"""
Runtime helpers for symmetric memory P2P allreduce in inductor-generated
Triton kernels.  Called from the generated wrapper code.
"""

from __future__ import annotations

from typing import Any

import torch


_symm_mem_cache: dict[str, Any] = {}


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
    key = group_name
    if key in _symm_mem_cache:
        return _symm_mem_cache[key]

    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem_mod

    workspace_bytes = input_tensor.numel() * input_tensor.element_size()
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

    result = (buf_ptrs, sig_ptrs, rank, world_size)
    _symm_mem_cache[key] = result
    return result
