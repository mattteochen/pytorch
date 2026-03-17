# Owner(s): ["oncall: distributed"]
"""
P2P allreduce op designed for fusion into inductor-generated Triton kernels.

Unlike ``fused_all_reduce_rmsnorm`` (the monolithic fused op), this op
**only** performs the allreduce.  The downstream compute (RMSNorm, add, etc.)
is left in the FX graph so that inductor's scheduler fuses it naturally into
the same Triton kernel.

When lowered by inductor, this op produces a ``SymmMemP2PAllReduce`` IR
Pointwise node whose inner_fn emits ``ops.symm_mem_p2p_reduce_load``.
The Triton codegen translates that into a loop of P2P loads from all peer
symmetric-memory buffers with kraken device-side synchronisation.

The ``CompositeExplicitAutograd`` fallback decomposes to standard NCCL
``all_reduce`` + ``wait_tensor`` so the op is safe everywhere.
"""

import torch
import torch.distributed._functional_collectives as funcol


lib = torch.library.Library("symm_mem", "FRAGMENT")

lib.define(
    "p2p_allreduce(Tensor input, str reduce_op, str group_name) -> Tensor",
    tags=[torch._C.Tag.needs_fixed_stride_order],
)


@torch.library.impl(lib, "p2p_allreduce", "Meta")
def _p2p_allreduce_meta(
    input: torch.Tensor,
    reduce_op: str,
    group_name: str,
) -> torch.Tensor:
    return torch.empty_like(input)


@torch.library.impl(lib, "p2p_allreduce", "CompositeExplicitAutograd")
def _p2p_allreduce_fallback(
    input: torch.Tensor,
    reduce_op: str,
    group_name: str,
) -> torch.Tensor:
    """Fallback: standard NCCL all_reduce + wait_tensor."""
    reduced = funcol.all_reduce(input, reduce_op, group_name)
    return reduced
