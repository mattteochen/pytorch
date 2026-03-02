"""
Standalone repro: upstream pointwise -> allreduce -> norm (Lamport mode).

Validates the MoE use case where a pointwise op (e.g. MoE output scaling)
feeds into allreduce + RMSNorm, all fused into a single Triton kernel.

Usage (requires 2+ GPUs with P2P access):
  torchrun --nproc_per_node=2 dev/moe/test_moe_allreduce_norm.py
"""

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.distributed._functional_collectives import all_reduce


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)
    device = torch.device("cuda", rank)

    hidden = 128
    rows = 8
    eps = 1e-5

    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    residual = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    weight = torch.ones(hidden, device=device, dtype=torch.float32)

    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)
    symm_mem.get_symm_mem_workspace(
        group_name, min_size=x.numel() * x.element_size()
    )

    # Reference
    upstream = x * 2.0
    ref_reduced = torch.ops._c10d_functional.all_reduce(upstream, "sum", group_name)
    ref_reduced = torch.ops._c10d_functional.wait_tensor(ref_reduced)
    ref_pre_norm = ref_reduced + residual
    ref_normed = F.rms_norm(ref_pre_norm, weight.shape, weight, eps)

    @torch.compile(options={
        "_fuse_symm_mem_comms": True,
        "_symm_mem_sync_mode": "lamport",
    })
    def upstream_ar_norm(inp, res, w, gn):
        up = inp * 2.0
        reduced = all_reduce(up, "sum", group=gn)
        h = reduced + res
        normed = F.rms_norm(h, w.shape, w, eps)
        return normed, h

    verbose = os.environ.get("TORCH_LOGS", "") != ""

    with torch.inference_mode():
        (normed, pre_norm), code = run_and_get_code(
            upstream_ar_norm, x, residual, weight, group_name
        )

    if verbose and rank == 0:
        print("=== Generated code ===")
        for c in code:
            print(c)

    torch.testing.assert_close(normed, ref_normed, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(pre_norm, ref_pre_norm, atol=2e-2, rtol=2e-2)

    # Verify Lamport codegen
    joined = "\n".join(code)
    assert "_lamport_poll_all_peers" in joined, "Missing Lamport poll"
    assert "_lamport_clear_old_slot" in joined, "Missing Lamport epilogue"

    if rank == 0:
        print("PASSED: upstream pointwise + allreduce + norm (Lamport mode)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
