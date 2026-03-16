"""
Lamport push-model P2P allreduce helpers for inductor-generated Triton kernels.

@triton.jit helpers for the sentinel protocol (push, volatile poll, fence,
clear) and Python workspace management (triple-buffered symmetric memory).

Key correctness invariant: the poll-load must check every element it reads for
the -0.0 sentinel before using the data.  The poll IS the data read — there is
no separate poll-then-load step.  This matches FlashInfer's methodology and
avoids relying on NVLink cache-line delivery order.
See triton_symm_mem.py for the full protocol description.
"""

from __future__ import annotations

from typing import Any

import torch

import triton
import triton.language as tl


NEG_ZERO_U16 = 0x8000
_NEG_ZERO = tl.constexpr(0x8000)


# ---------------------------------------------------------------------------
# Triton JIT helpers (primitives)
# ---------------------------------------------------------------------------


@triton.jit
def _fence_sys():
    tl.inline_asm_elementwise(
        "fence.sc.sys;",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _fence_acquire_sys():
    tl.inline_asm_elementwise(
        "fence.acquire.sys;",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _volatile_load_u32_scalar(addr):
    return tl.inline_asm_elementwise(
        "ld.volatile.global.b32 $0, [$1];",
        "=r, l",
        [addr],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _remove_neg_zero(val):
    bits = val.to(tl.uint16, bitcast=True)
    return tl.where(bits == _NEG_ZERO, tl.zeros_like(val), val)


@triton.jit
def _has_neg_zero(val):
    """Return scalar int32 1 if any element in val is -0.0, else 0."""
    bits = val.to(tl.uint16, bitcast=True)
    has_sentinel = (bits == _NEG_ZERO).to(tl.int32)
    return (tl.sum(has_sentinel) > 0).to(tl.int32)


# ---------------------------------------------------------------------------
# Triton JIT helpers (composite operations, called from generated code)
# ---------------------------------------------------------------------------


@triton.jit
def _lamport_poll_load(
    buf_ptr,
    idx,
    mask,
    BLOCK_N: tl.constexpr,
):
    """Spin-load until all elements are non-sentinel, then return the data.

    The poll IS the read: we load the data, check every element for -0.0,
    and retry until the entire vector is clean.  This matches FlashInfer's
    methodology — no separate poll + load steps, no assumptions about
    NVLink cache-line delivery order.

    volatile=True prevents Triton from hoisting/CSE-ing the load out of
    the spin loop.
    """
    val = tl.load(buf_ptr + idx, mask=mask, other=0.0, volatile=True)
    has_sentinel = _has_neg_zero(val)
    while has_sentinel == 1:
        val = tl.load(buf_ptr + idx, mask=mask, other=0.0, volatile=True)
        has_sentinel = _has_neg_zero(val)
    return val


@triton.jit
def _lamport_clear_old_slot(
    clear_base,
    row_offset,
    cols,
    mask,
    chunk,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    neg_zero = tl.full([BLOCK_N], _NEG_ZERO, dtype=tl.uint16).to(
        tl.bfloat16, bitcast=True
    )
    for peer in tl.static_range(WORLD_SIZE):
        if peer != RANK:
            clear_bf16 = (clear_base + peer * chunk + row_offset).to(
                tl.pointer_type(tl.bfloat16)
            )
            tl.store(clear_bf16 + cols, neg_zero, mask=mask)


# ---------------------------------------------------------------------------
# In-kernel flag advancement (FlashInfer-style, zero wrapper GPU ops)
# ---------------------------------------------------------------------------


@triton.jit
def _lamport_block_arrive(meta_i32_ptr):
    tl.debug_barrier()
    tl.atomic_add(meta_i32_ptr, 1, sem="release", scope="gpu")


@triton.jit
def _lamport_advance_flag_block0(meta_i32_ptr, flag_value):
    # Block 0: spin on meta[0] until all blocks arrived, advance meta[1], reset.
    if tl.program_id(0) == 0:
        _lam_expected = tl.num_programs(0)
        _lam_ready = tl.full([], 0, dtype=tl.int32)
        while _lam_ready == 0:
            _lam_cnt = _volatile_load_u32_scalar(meta_i32_ptr)
            _lam_ready = (_lam_cnt == _lam_expected).to(tl.int32)
        tl.store(meta_i32_ptr + 1, (flag_value + 1) % 3)
        tl.store(meta_i32_ptr, tl.full([], 0, dtype=tl.int32))


# ---------------------------------------------------------------------------
# Python runtime helpers (called from generated wrapper code)
# ---------------------------------------------------------------------------

_lamport_cache: dict[str, Any] = {}


def lamport_workspace_setup(
    input_tensor: torch.Tensor,
    group_name: str,
) -> tuple[torch.Tensor, int, int, torch.Tensor]:
    """Allocate (or retrieve cached) triple-buffered workspace + metadata.

    Returns (buf_ptrs, rank, world_size, meta_tensor).
    meta is int32[2]: [block_counter, flag_value]. Updated in-kernel only.
    """
    key = group_name
    if key in _lamport_cache:
        cached = _lamport_cache[key]
        return (
            cached["buf_ptrs"],
            cached["rank"],
            cached["world_size"],
            cached["meta"],
        )

    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem_mod

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    M, N = input_tensor.shape[-2], input_tensor.shape[-1]
    if N % 2 != 0:
        raise ValueError(
            f"Lamport allreduce requires even reduction dim, got {N}. "
            "The sentinel protocol uses bf16 -0.0 (0x8000); "
            "an odd dim would leave a dangling byte."
        )
    device = input_tensor.device

    slot_elems = world_size * M * N
    total_elems = 3 * slot_elems

    buf = symm_mem_mod.empty(total_elems, dtype=torch.bfloat16, device=device)
    sm = symm_mem_mod.rendezvous(buf, group_name)
    buf.view(torch.uint16).fill_(NEG_ZERO_U16)
    sm.barrier(channel=0)

    buf_ptrs = torch.tensor(
        [sm.buffer_ptrs[i] for i in range(world_size)],
        dtype=torch.int64,
        device=device,
    )

    # GPU-resident metadata: [block_counter, flag_value], both start at 0.
    meta = torch.zeros(2, dtype=torch.int32, device=device)

    _lamport_cache[key] = {
        "sm": sm,
        "buf": buf,
        "buf_ptrs": buf_ptrs,
        "rank": rank,
        "world_size": world_size,
        "meta": meta,
    }
    return (buf_ptrs, rank, world_size, meta)


def lamport_workspace_peer_bufs(
    input_tensor: torch.Tensor,
    group_name: str,
) -> tuple[list[torch.Tensor], int, int, torch.Tensor]:
    """Return per-peer tensor views of the triple-buffered workspace."""
    _buf_ptrs, rank, world_size, meta = lamport_workspace_setup(
        input_tensor, group_name
    )
    cached = _lamport_cache[group_name]
    sm = cached["sm"]
    total_elems = cached["buf"].numel()
    peer_bufs = [
        sm.get_buffer(i, (total_elems,), torch.bfloat16)
        for i in range(world_size)
    ]
    return peer_bufs, rank, world_size, meta
