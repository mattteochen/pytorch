"""
Runtime helpers for NVSHMEM-based P2P allreduce in inductor-generated
Triton kernels.  Called from the generated wrapper code.

Mirrors symm_mem_helpers.py but adds epoch tracking for NVSHMEM signals.
The epoch monotonically increases across invocations so signals never need
resetting — critical for CUDA graph replay.
"""

from __future__ import annotations

from typing import Any

import torch


_nvshmem_cache: dict[str, Any] = {}
_nvshmem_epoch: dict[str, int] = {}


def _get_cached(
    input_tensor: torch.Tensor,
    group_name: str,
) -> tuple[Any, torch.Tensor, torch.Tensor, int, int]:
    """Return ``(sm_handle, buf_ptrs, sig_ptrs, rank, world_size)``, cached."""
    key = group_name
    workspace_bytes = input_tensor.numel() * input_tensor.element_size()
    if key in _nvshmem_cache:
        cached = _nvshmem_cache[key]
        sm = cached[0]
        if sm.buffer_size >= workspace_bytes:
            return cached
        del _nvshmem_cache[key]

    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem_mod

    sm = symm_mem_mod.get_symm_mem_workspace(group_name, min_size=workspace_bytes)

    rank = dist.get_rank()
    world_size = sm.world_size
    device = input_tensor.device

    buf_ptrs = torch.tensor(
        [sm.buffer_ptrs[i] for i in range(world_size)],
        dtype=torch.int64,
        device=device,
    )
    sig_ptrs = torch.tensor(
        [sm.signal_pad_ptrs[i] for i in range(world_size)],
        dtype=torch.int64,
        device=device,
    )

    result = (sm, buf_ptrs, sig_ptrs, rank, world_size)
    _nvshmem_cache[key] = result
    return result


def nvshmem_peer_bufs(
    input_tensor: torch.Tensor,
    group_name: str,
) -> tuple[list[torch.Tensor], torch.Tensor, int, int]:
    """
    Return per-peer tensor views and signal pad pointers for NVSHMEM mode.

    Returns ``(peer_bufs, signal_pad_ptrs_tensor, rank, world_size)``.
    """
    sm, _buf_ptrs, sig_ptrs, rank, world_size = _get_cached(
        input_tensor, group_name
    )
    shape = input_tensor.shape
    dtype = input_tensor.dtype
    peer_bufs = [sm.get_buffer(i, shape, dtype) for i in range(world_size)]
    return peer_bufs, sig_ptrs, rank, world_size


def nvshmem_get_epoch(group_name: str) -> int:
    """Return the current epoch for *group_name*, then increment."""
    epoch = _nvshmem_epoch.get(group_name, 1)
    _nvshmem_epoch[group_name] = epoch + 1
    return epoch
