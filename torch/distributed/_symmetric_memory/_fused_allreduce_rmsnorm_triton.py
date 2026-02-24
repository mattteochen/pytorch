# Owner(s): ["oncall: distributed"]
"""
Fused all_reduce + RMSNorm Triton kernel using symmetric memory P2P.

Single kernel performs: P2P loads from all peers → sum reduce → optional
residual add → RMSNorm.  Host-side ``symm_mem.barrier()`` provides
synchronization before and after the kernel.  No NVSHMEM dependency.
"""

import torch
import triton
import triton.language as tl
from torch.distributed._symmetric_memory import get_symm_mem_workspace

_MAX_WORLD_SIZE = 8


@triton.jit
def _fused_allreduce_rmsnorm_kernel(
    # P2P buffer pointers (one per rank, unused ranks are ignored)
    buf0,
    buf1,
    buf2,
    buf3,
    buf4,
    buf5,
    buf6,
    buf7,
    output_ptr,
    weight_ptr,
    residual_ptr,
    residual_out_ptr,
    N,
    stride_row,
    eps,
    WORLD_SIZE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    row_off = row_idx * stride_row + cols

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if WORLD_SIZE >= 1:
        acc += tl.load(buf0 + row_off, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 2:
        acc += tl.load(buf1 + row_off, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 3:
        acc += tl.load(buf2 + row_off, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 4:
        acc += tl.load(buf3 + row_off, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 5:
        acc += tl.load(buf4 + row_off, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 6:
        acc += tl.load(buf5 + row_off, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 7:
        acc += tl.load(buf6 + row_off, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 8:
        acc += tl.load(buf7 + row_off, mask=mask, other=0.0).to(tl.float32)

    if HAS_RESIDUAL:
        res = tl.load(residual_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        acc = acc + res
        tl.store(
            residual_out_ptr + row_off,
            acc.to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    mean_sq = tl.sum(acc * acc, axis=0) / N
    rnorm = tl.math.rsqrt(mean_sq + eps)

    wt = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = acc * rnorm * wt

    tl.store(
        output_ptr + row_off,
        out.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def _launch_fused_kernel(
    sm,
    peer_bufs: list[torch.Tensor],
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    *,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """barrier → Triton kernel → barrier.

    ``peer_bufs`` must already contain each rank's data (either via explicit
    copy or because the tensors were allocated in symmetric memory).
    """
    M, N = input_2d.shape
    output = torch.empty_like(input_2d)
    has_residual = residual is not None

    residual_2d: torch.Tensor | None = None
    residual_out: torch.Tensor | None = None
    if has_residual:
        assert residual is not None
        residual_2d = residual.reshape(-1, residual.shape[-1])
        residual_out = torch.empty_like(input_2d)

    BLOCK_N = triton.next_power_of_2(N)

    sm.barrier(channel=0)

    _fused_allreduce_rmsnorm_kernel[(M,)](
        peer_bufs[0],
        peer_bufs[1],
        peer_bufs[2],
        peer_bufs[3],
        peer_bufs[4],
        peer_bufs[5],
        peer_bufs[6],
        peer_bufs[7],
        output,
        weight,
        residual_2d if has_residual else output,
        residual_out if has_residual else output,
        N,
        N,
        eps,
        WORLD_SIZE=sm.world_size,
        HAS_RESIDUAL=has_residual,
        BLOCK_N=BLOCK_N,
    )

    sm.barrier(channel=0)
    return output, residual_out


def _make_peer_bufs(sm, shape: tuple[int, ...], dtype: torch.dtype) -> list[torch.Tensor]:
    """Build the padded peer buffer list from a SymmetricMemory handle."""
    peer_bufs = [sm.get_buffer(r, shape, dtype) for r in range(sm.world_size)]
    while len(peer_bufs) < _MAX_WORLD_SIZE:
        peer_bufs.append(peer_bufs[0])
    return peer_bufs


def fused_allreduce_rmsnorm_symm_mem(
    input: torch.Tensor,
    weight: torch.Tensor,
    reduce_op: str,
    group_name: str,
    *,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if reduce_op != "sum":
        raise ValueError(f"Only 'sum' reduce_op is supported, got '{reduce_op}'")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    if residual is not None and not residual.is_contiguous():
        raise ValueError("residual must be contiguous")

    input_2d = input.reshape(-1, input.shape[-1])
    workspace_bytes = input_2d.numel() * input_2d.element_size()

    sm = get_symm_mem_workspace(group_name, min_size=workspace_bytes)
    if sm.world_size > _MAX_WORLD_SIZE:
        raise ValueError(
            f"world_size {sm.world_size} exceeds maximum {_MAX_WORLD_SIZE}"
        )

    peer_bufs = _make_peer_bufs(sm, tuple(input_2d.shape), input_2d.dtype)
    peer_bufs[sm.rank].copy_(input_2d)

    output, residual_out = _launch_fused_kernel(
        sm, peer_bufs, input_2d, weight, residual=residual, eps=eps,
    )

    output = output.view(input.shape)
    if residual_out is not None:
        residual_out = residual_out.view(input.shape)

    return output, residual_out
