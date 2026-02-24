"""
Runtime helpers for symmetric memory P2P allreduce in inductor-generated
Triton kernels.  Called from the generated wrapper code.
"""

from __future__ import annotations

from typing import Any

import torch


_symm_mem_cache: dict[str, Any] = {}


def _get_cached(
    input_tensor: torch.Tensor,
    group_name: str,
) -> tuple[Any, torch.Tensor, torch.Tensor, int, int]:
    """Return ``(sm_handle, buf_ptrs, sig_ptrs, rank, world_size)``, cached."""
    key = group_name
    workspace_bytes = input_tensor.numel() * input_tensor.element_size()
    if key in _symm_mem_cache:
        cached = _symm_mem_cache[key]
        sm = cached[0]
        cached_bytes = sm.buffer_size
        assert cached_bytes >= workspace_bytes, (
            f"Cached symmetric memory workspace for group '{group_name}' is too "
            f"small ({cached_bytes} bytes) for input ({workspace_bytes} bytes). "
            f"Allocate a larger workspace with get_symm_mem_workspace(min_size=...)."
        )
        return cached

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
    _symm_mem_cache[key] = result
    return result


def symm_mem_setup(
    input_tensor: torch.Tensor,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Return cached symm-mem pointer tensors for *group_name*.

    The first call performs the workspace allocation and rendezvous (a
    collective).  Subsequent calls return cached CUDA int64 tensors.

    Returns ``(buf_ptrs_tensor, signal_pad_ptrs_tensor, rank, world_size)``.
    """
    _sm, buf_ptrs, sig_ptrs, rank, world_size = _get_cached(input_tensor, group_name)
    return (buf_ptrs, sig_ptrs, rank, world_size)


def symm_mem_host_barrier_setup(
    input_tensor: torch.Tensor,
    group_name: str,
    skip_copy: bool = False,
) -> tuple[torch.Tensor, int, int]:
    """
    Pre-kernel setup for host-barrier mode.

    Copies *input_tensor* into the symmetric memory workspace (unless
    *skip_copy*).  The caller is responsible for issuing
    ``symm_mem_host_barrier`` before and after the kernel.

    Returns ``(buf_ptrs_tensor, rank, world_size)`` for the kernel call.
    """
    sm, buf_ptrs, _sig_ptrs, rank, world_size = _get_cached(input_tensor, group_name)

    if not skip_copy:
        local_buf = sm.get_buffer(rank, input_tensor.shape, input_tensor.dtype)
        local_buf.copy_(input_tensor)

    return (buf_ptrs, rank, world_size)


def symm_mem_host_barrier(
    group_name: str,
) -> None:
    """Post-kernel host-side barrier using the cached SymmetricMemory handle."""
    sm = _symm_mem_cache[group_name][0]
    sm.barrier(channel=0)
