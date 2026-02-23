"""
Runtime helpers for Lamport push-model P2P allreduce in inductor-generated
Triton kernels.

Provides:
  - @triton.jit helper functions for the Lamport sentinel protocol
    (push to peers, volatile poll, fence, sentinel clear)
  - Python workspace management (triple-buffered symmetric memory)
  - Iteration tracking for buffer rotation

The codegen emits imports from this module and calls the JIT helpers
inside the generated kernel, plus the Python helpers in the wrapper.
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
    """System-scope fence ensuring all prior stores are visible to all GPUs."""
    tl.inline_asm_elementwise(
        "fence.sc.sys;",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _volatile_load_u32_scalar(addr):
    """Single scalar volatile load bypassing L1 cache."""
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
    """Replace bf16 -0.0 with +0.0 so real data never matches sentinel."""
    bits = val.to(tl.uint16, bitcast=True)
    return tl.where(bits == _NEG_ZERO, tl.zeros_like(val), val)


@triton.jit
def _poll_last_word(slot_u32_ptr, n_words):
    """Spin on the last u32 word of a slot until it contains no sentinel.

    Since the writer stores sequentially with a system fence, if the
    last word is ready, all prior words are guaranteed visible.
    """
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
    buf_ptrs,
    data,
    row_offset,
    cols,
    mask,
    chunk,
    buf_offset,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Push local data to all peers' symmetric memory buffers."""
    buf_ptrs_u64 = buf_ptrs.to(tl.pointer_type(tl.uint64))
    for peer in tl.static_range(WORLD_SIZE):
        peer_buf = tl.load(buf_ptrs_u64 + peer).to(tl.pointer_type(tl.bfloat16))
        tl.store(
            peer_buf + buf_offset + RANK * chunk + row_offset + cols,
            data,
            mask=mask,
        )


@triton.jit
def _lamport_poll_and_reduce(
    buf_ptrs,
    row_offset,
    cols,
    mask,
    chunk,
    n_words,
    buf_offset,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Poll own local buffer for all peers' data, accumulate into fp32."""
    buf_ptrs_u64 = buf_ptrs.to(tl.pointer_type(tl.uint64))
    my_buf = tl.load(buf_ptrs_u64 + RANK).to(tl.pointer_type(tl.bfloat16))
    my_buf_base = my_buf + buf_offset

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for peer in tl.static_range(WORLD_SIZE):
        slot_bf16 = my_buf_base + peer * chunk + row_offset
        slot_u32 = slot_bf16.to(tl.pointer_type(tl.uint32))
        _poll_last_word(slot_u32, n_words)
        val = tl.load(slot_bf16 + cols, mask=mask, other=0.0)
        acc += val.to(tl.float32)
    return acc


@triton.jit
def _lamport_clear_old_slot(
    buf_ptrs,
    row_offset,
    cols,
    mask,
    chunk,
    clear_offset,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Write -0.0 sentinels to the old buffer slot (from 2 iterations ago)."""
    buf_ptrs_u64 = buf_ptrs.to(tl.pointer_type(tl.uint64))
    my_buf = tl.load(buf_ptrs_u64 + RANK).to(tl.pointer_type(tl.bfloat16))
    clear_base = my_buf + clear_offset
    neg_zero = tl.full([BLOCK_N], _NEG_ZERO, dtype=tl.uint16).to(
        tl.bfloat16, bitcast=True
    )
    for peer in tl.static_range(WORLD_SIZE):
        clear_bf16 = (clear_base + peer * chunk + row_offset).to(
            tl.pointer_type(tl.bfloat16)
        )
        tl.store(clear_bf16 + cols, neg_zero, mask=mask)


# ---------------------------------------------------------------------------
# Python runtime helpers (called from generated wrapper code)
# ---------------------------------------------------------------------------

_lamport_cache: dict[str, Any] = {}


def lamport_workspace_setup(
    input_tensor: torch.Tensor,
    group_name: str,
) -> tuple[torch.Tensor, int, int, torch.Tensor]:
    """
    Allocate (or retrieve cached) triple-buffered symmetric memory workspace
    and a GPU-resident offsets tensor for CUDA-graph-safe buffer rotation.

    Returns ``(buf_ptrs, rank, world_size, offsets_tensor)``.

    ``offsets_tensor`` is a ``[2]`` int64 CUDA tensor holding
    ``[buf_offset, clear_offset]``.  Call :func:`lamport_advance_offsets`
    before each kernel launch to rotate the triple buffer -- that call is
    a pure GPU op and is safely captured in CUDA graphs.
    """
    key = group_name
    if key in _lamport_cache:
        cached = _lamport_cache[key]
        return (
            cached["buf_ptrs"],
            cached["rank"],
            cached["world_size"],
            cached["offsets"],
        )

    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem_mod

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    M, N = input_tensor.shape[-2], input_tensor.shape[-1]
    device = input_tensor.device

    slot_elems = world_size * M * N
    total_elems = 3 * slot_elems

    buf = symm_mem_mod.empty(total_elems, dtype=torch.bfloat16, device=device)
    sm = symm_mem_mod.rendezvous(buf, dist.group.WORLD)
    buf.view(torch.uint16).fill_(NEG_ZERO_U16)
    sm.barrier(channel=0)

    buf_ptrs = torch.tensor(
        [sm.buffer_ptrs[i] for i in range(world_size)],
        dtype=torch.int64,
        device=device,
    )

    # GPU-resident counter (scalar) and offsets ([2]) for CUDA graph safety.
    # The counter starts at -1 so the first advance_offsets brings it to 0.
    counter = torch.tensor(-1, dtype=torch.int64, device=device)
    offsets = torch.zeros(2, dtype=torch.int64, device=device)

    _lamport_cache[key] = {
        "sm": sm,
        "buf": buf,
        "buf_ptrs": buf_ptrs,
        "rank": rank,
        "world_size": world_size,
        "slot_elems": slot_elems,
        "counter": counter,
        "offsets": offsets,
    }
    return (buf_ptrs, rank, world_size, offsets)


def lamport_advance_offsets(group_name: str) -> None:
    """
    Advance the triple-buffer rotation counter and update the offsets tensor.

    Uses only element-wise arithmetic (add, remainder, mul) on scalar
    GPU tensors -- no fancy indexing or data-dependent dispatch -- so this
    is safely captured in CUDA graphs and replays correctly.
    """
    entry = _lamport_cache[group_name]
    counter = entry["counter"]
    offsets = entry["offsets"]
    slot_elems = entry["slot_elems"]

    counter.add_(1)
    # buf_offset  = (counter % 3) * slot_elems
    # clear_offset = ((counter + 2) % 3) * slot_elems
    offsets[0] = (counter % 3) * slot_elems
    offsets[1] = ((counter + 2) % 3) * slot_elems
