# Owner(s): ["oncall: distributed"]
"""
Triton kernel for fused all_reduce + RMSNorm using P2P symmetric memory.

The kernel reads directly from all ranks' P2P-mapped symmetric memory buffers,
reduces in-register, optionally adds a residual, and computes RMS normalization
-- all in a single kernel launch. This avoids materializing the allreduce
intermediate in global memory.

The host wrapper handles symmetric memory allocation, local data staging, and
cross-rank barrier synchronization. A future version can move the barriers
into the kernel using NVSHMEM cooperative primitives when Triton adds
cooperative launch support.
"""

import torch
import triton
import triton.language as tl

from torch.distributed._symmetric_memory import (
    _get_backend_stream,
    get_symm_mem_workspace,
)

_MAX_WORLD_SIZE = 8


@triton.jit
def _fused_allreduce_rmsnorm_kernel(
    buf0_ptr,
    buf1_ptr,
    buf2_ptr,
    buf3_ptr,
    buf4_ptr,
    buf5_ptr,
    buf6_ptr,
    buf7_ptr,
    output_ptr,
    weight_ptr,
    residual_ptr,
    residual_out_ptr,
    N,
    stride_buf_row,
    stride_out_row,
    stride_res_row,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    buf_offset = row_idx * stride_buf_row + cols

    acc = tl.load(buf0_ptr + buf_offset, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 2:
        acc += tl.load(buf1_ptr + buf_offset, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 3:
        acc += tl.load(buf2_ptr + buf_offset, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 4:
        acc += tl.load(buf3_ptr + buf_offset, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 5:
        acc += tl.load(buf4_ptr + buf_offset, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 6:
        acc += tl.load(buf5_ptr + buf_offset, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 7:
        acc += tl.load(buf6_ptr + buf_offset, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 8:
        acc += tl.load(buf7_ptr + buf_offset, mask=mask, other=0.0).to(tl.float32)

    if HAS_RESIDUAL:
        res = tl.load(
            residual_ptr + row_idx * stride_res_row + cols, mask=mask, other=0.0
        )
        pre_norm = acc + res.to(tl.float32)
        tl.store(
            residual_out_ptr + row_idx * stride_out_row + cols,
            pre_norm.to(output_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        pre_norm = acc

    mean_sq = tl.sum(pre_norm * pre_norm, axis=0) / N
    rnorm = tl.math.rsqrt(mean_sq + eps)

    wt = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    out = pre_norm * rnorm * wt.to(tl.float32)

    tl.store(
        output_ptr + row_idx * stride_out_row + cols,
        out.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_allreduce_rmsnorm_nvshmem(
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

    input_2d = input.reshape(-1, input.shape[-1])
    M, N = input_2d.shape

    workspace_bytes = input_2d.numel() * input_2d.element_size()
    symm_mem = get_symm_mem_workspace(group_name, min_size=workspace_bytes)
    world_size = symm_mem.world_size
    rank = symm_mem.rank

    if world_size > _MAX_WORLD_SIZE:
        raise ValueError(
            f"world_size {world_size} exceeds maximum {_MAX_WORLD_SIZE}"
        )

    local_buf = symm_mem.get_buffer(rank, input_2d.shape, input_2d.dtype)

    backend_stream = _get_backend_stream()
    backend_stream.wait_stream(torch.cuda.current_stream())
    with backend_stream:
        local_buf.copy_(input_2d)

    symm_mem.barrier(channel=0)
    torch.cuda.current_stream().wait_stream(backend_stream)

    bufs = [symm_mem.get_buffer(r, input_2d.shape, input_2d.dtype) for r in range(world_size)]
    while len(bufs) < _MAX_WORLD_SIZE:
        bufs.append(bufs[0])

    output = torch.empty_like(input_2d)
    has_residual = residual is not None

    residual_2d: torch.Tensor | None = None
    residual_out: torch.Tensor | None = None
    if has_residual:
        assert residual is not None
        residual_2d = residual.reshape(-1, residual.shape[-1])
        residual_out = torch.empty_like(input_2d)

    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)

    _fused_allreduce_rmsnorm_kernel[grid](
        bufs[0], bufs[1], bufs[2], bufs[3],
        bufs[4], bufs[5], bufs[6], bufs[7],
        output,
        weight,
        residual_2d if has_residual else output,
        residual_out if has_residual else output,
        N,
        N,  # stride_buf_row (contiguous)
        N,  # stride_out_row (contiguous)
        residual_2d.stride(0) if residual_2d is not None else N,
        eps,
        HAS_RESIDUAL=has_residual,
        WORLD_SIZE=world_size,
        BLOCK_N=BLOCK_N,
    )

    symm_mem.barrier(channel=0)

    output = output.view(input.shape)
    if residual_out is not None:
        residual_out = residual_out.view(input.shape)

    return output, residual_out
