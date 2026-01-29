# Owner(s): ["oncall: distributed"]
"""
Fused all_reduce + RMSNorm operation using symmetric memory.

This module defines the fused_all_reduce_rmsnorm op which performs:
1. All-reduce across ranks
2. Optional residual addition
3. RMS normalization

All in a single kernel to avoid intermediate memory round-trips.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


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
    ") -> Tensor",
    tags=[torch._C.Tag.needs_fixed_stride_order],
)


@torch.library.impl(lib, "fused_all_reduce_rmsnorm", "Meta")
def _fused_all_reduce_rmsnorm_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    reduce_op: str,
    group_name: str,
    *,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Meta implementation for shape inference."""
    return torch.empty_like(input)


@torch.library.impl(lib, "fused_all_reduce_rmsnorm", "CompositeExplicitAutograd")
def _fused_all_reduce_rmsnorm_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    reduce_op: str,
    group_name: str,
    *,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fallback implementation using existing ops.

    This is used for:
    - Eager execution when symmetric memory is not available
    - Tracing/compilation to understand the semantics
    - Testing correctness
    """
    # Step 1: All-reduce
    reduced = torch.ops._c10d_functional.all_reduce(input, reduce_op, group_name)
    reduced = torch.ops._c10d_functional.wait_tensor(reduced)

    # Step 2: Optional residual add
    if residual is not None:
        reduced = reduced + residual

    # Step 3: RMS normalization
    normalized_shape = weight.shape
    return F.rms_norm(reduced, normalized_shape, weight, eps)


# CUDA implementation will be added when the Triton kernel is ready
# @torch.library.impl(lib, "fused_all_reduce_rmsnorm", "CUDA")
# def _fused_all_reduce_rmsnorm_cuda(...):
#     """
#     Optimized CUDA implementation using symmetric memory.
#     Uses a single Triton kernel that:
#     1. Reads from symmetric memory buffers across ranks
#     2. Performs all-reduce via direct memory access
#     3. Computes RMS normalization
#     4. Writes output in a single pass
#     """
#     pass
