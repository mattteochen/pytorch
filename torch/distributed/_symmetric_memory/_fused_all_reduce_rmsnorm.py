# Owner(s): ["oncall: distributed"]
"""
Fused all_reduce + RMSNorm operation using symmetric memory.

This module defines the fused_all_reduce_rmsnorm op which performs:
1. All-reduce across ranks
2. Optional residual addition
3. RMS normalization

All in a single kernel to avoid intermediate memory round-trips.

Returns (normed_output, pre_norm_output) where pre_norm_output is the
reduced + residual value (useful as the next layer's residual input).
pre_norm_output is None when no residual is provided.
"""

import logging

import torch
import torch.nn.functional as F


log = logging.getLogger(__name__)

# Define the custom op in the symm_mem namespace
lib = torch.library.Library("symm_mem", "FRAGMENT")

lib.define(
    "fused_all_reduce_rmsnorm("
    "Tensor input, "
    "Tensor weight, "
    "str reduce_op, "
    "str group_name, "
    "*, "
    "Tensor? residual = None, "
    "float eps = 1e-6"
    ") -> (Tensor, Tensor?)",
    tags=[torch._C.Tag.needs_fixed_stride_order],
)


@torch.library.impl(lib, "fused_all_reduce_rmsnorm", "Meta")
def _fused_all_reduce_rmsnorm_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    reduce_op: str,
    group_name: str,
    *,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    normed = torch.empty_like(input)
    pre_norm = torch.empty_like(input) if residual is not None else None
    return normed, pre_norm


@torch.library.impl(lib, "fused_all_reduce_rmsnorm", "CompositeExplicitAutograd")
def _fused_all_reduce_rmsnorm_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    reduce_op: str,
    group_name: str,
    *,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Fallback implementation using existing ops.

    Used for eager execution, tracing, and correctness testing.
    """
    reduced = torch.ops._c10d_functional.all_reduce(input, reduce_op, group_name)
    reduced = torch.ops._c10d_functional.wait_tensor(reduced)

    pre_norm: torch.Tensor | None = None
    if residual is not None:
        reduced = reduced + residual
        pre_norm = reduced

    normed = F.rms_norm(reduced, weight.shape, weight, eps)
    return normed, pre_norm


def _is_nvshmem_available() -> bool:
    try:
        from torch.distributed._symmetric_memory._nvshmem_triton import (
            NvshmemLibFinder,
        )

        NvshmemLibFinder.find_device_library()
        return True
    except (ImportError, RuntimeError):
        return False


@torch.library.impl(lib, "fused_all_reduce_rmsnorm", "CUDA")
def _fused_all_reduce_rmsnorm_cuda(
    input: torch.Tensor,
    weight: torch.Tensor,
    reduce_op: str,
    group_name: str,
    *,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if _is_nvshmem_available():
        try:
            from ._fused_allreduce_rmsnorm_triton import (
                fused_allreduce_rmsnorm_nvshmem,
            )

            return fused_allreduce_rmsnorm_nvshmem(
                input, weight, reduce_op, group_name, residual=residual, eps=eps
            )
        except (ImportError, RuntimeError, ValueError):
            log.debug(
                "NVSHMEM Triton kernel failed, falling back to decomposed impl",
                exc_info=True,
            )

    return _fused_all_reduce_rmsnorm_fallback(
        input, weight, reduce_op, group_name, residual=residual, eps=eps
    )
