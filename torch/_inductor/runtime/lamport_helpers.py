"""
Lamport push-model P2P allreduce helpers for inductor-generated Triton kernels.

@triton.jit helpers for the sentinel protocol (push, volatile poll, fence,
clear) and Python workspace management (triple-buffered symmetric memory).
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
    # Pairs with writer's fence.sc.sys; needed after sentinel poll before
    # non-volatile data loads to prevent stale L2 cache hits.
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
def _poll_last_word(slot_u32_ptr, n_words):
    last_addr = slot_u32_ptr + (n_words - 1)
    ready = tl.full([], 0, dtype=tl.int32)
    while ready == 0:
        w = _volatile_load_u32_scalar(last_addr)
        lo = w & 0xFFFF
        hi = (w >> 16) & 0xFFFF
        ready = ((lo != _NEG_ZERO) & (hi != _NEG_ZERO)).to(tl.int32)


# ---------------------------------------------------------------------------
# Triton JIT helpers (composite operations, called from generated code)
# ---------------------------------------------------------------------------


@triton.jit
def _lamport_push_to_peers(
    buf_ptrs_u64,
    data,
    row_offset,
    cols,
    mask,
    chunk,
    buf_offset,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    for peer in tl.static_range(WORLD_SIZE):
        if peer != RANK:
            peer_buf = tl.load(buf_ptrs_u64 + peer).to(tl.pointer_type(tl.bfloat16))
            tl.store(
                peer_buf + buf_offset + RANK * chunk + row_offset + cols,
                data,
                mask=mask,
            )


@triton.jit
def _lamport_poll_and_reduce(
    my_buf_base,
    row_offset,
    cols,
    mask,
    chunk,
    n_words,
    WORLD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for peer in tl.static_range(WORLD_SIZE):
        slot_bf16 = my_buf_base + peer * chunk + row_offset
        slot_u32 = slot_bf16.to(tl.pointer_type(tl.uint32))
        _poll_last_word(slot_u32, n_words)
        val = tl.load(slot_bf16 + cols, mask=mask, other=0.0)
        acc += val.to(tl.float32)
    return acc


@triton.jit
def _lamport_poll_rows(
    my_buf_base,
    x_base,
    r0_numel,
    chunk,
    n_words,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    XBLOCK: tl.constexpr,
    xnumel,
):
    for row in tl.static_range(XBLOCK):
        row_idx = x_base + row
        if row_idx < xnumel:
            row_offset = row_idx * r0_numel
            for peer in tl.static_range(WORLD_SIZE):
                if peer != RANK:
                    slot_bf16 = my_buf_base + peer * chunk + row_offset
                    slot_u32 = slot_bf16.to(tl.pointer_type(tl.uint32))
                    _poll_last_word(slot_u32, n_words)
    _fence_acquire_sys()


@triton.jit
def _lamport_poll_all_peers(
    my_buf_base,
    x_base,
    r0_numel,
    chunk,
    n_words,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    XBLOCK: tl.constexpr,
    xnumel,
):
    # Simultaneous poll: single while-loop checks all non-self peers per row.
    for row in tl.static_range(XBLOCK):
        row_idx = x_base + row
        if row_idx < xnumel:
            row_offset = row_idx * r0_numel
            _lam_done = tl.full([], 0, dtype=tl.int32)
            while _lam_done == 0:
                _lam_cnt = tl.full([], 0, dtype=tl.int32)
                for peer in tl.static_range(WORLD_SIZE):
                    if peer != RANK:
                        slot_bf16 = my_buf_base + peer * chunk + row_offset
                        slot_u32 = slot_bf16.to(tl.pointer_type(tl.uint32))
                        last_addr = slot_u32 + (n_words - 1)
                        w = _volatile_load_u32_scalar(last_addr)
                        lo = w & 0xFFFF
                        hi = (w >> 16) & 0xFFFF
                        peer_ready = ((lo != _NEG_ZERO) & (hi != _NEG_ZERO)).to(
                            tl.int32
                        )
                        _lam_cnt = _lam_cnt + peer_ready
                _lam_done = (_lam_cnt == WORLD_SIZE - 1).to(tl.int32)
    _fence_acquire_sys()


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
            "The sentinel protocol packs 2 bf16 elements per u32 word; "
            "an odd dim causes _poll_last_word to deadlock."
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
