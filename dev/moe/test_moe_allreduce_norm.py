"""
End-to-end repro: MoE (native grouped_mm v2) → allreduce → RMSNorm.

Compares:
  1. Eager reference: MoE eager → functional all_reduce + wait → rmsnorm
  2. Compiled + fused Lamport: MoE output feeds into a compiled function
     that does allreduce → residual add → rmsnorm with symm_mem fusion.

The compiled path exercises the upstream-pointwise + allreduce + norm
fusion because the MoE output flows directly into the allreduce kernel.

Usage (requires 2+ GPUs with P2P access):
  torchrun --nproc_per_node=2 dev/moe/test_moe_allreduce_norm.py
  TORCH_LOGS=output_code torchrun --nproc_per_node=2 dev/moe/test_moe_allreduce_norm.py
"""

import os
import signal
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.distributed._functional_collectives import all_reduce


# ── gpt-oss-20b model constants (scaled down for test) ──────────────────────
NUM_EXPERTS = 32
TOP_K = 4
HIDDEN_SIZE = 2880
INTERMEDIATE_SIZE = 2880
GEMM1_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
DTYPE = torch.bfloat16


# ── Activation ──────────────────────────────────────────────────────────────
def swiglu_with_alpha_and_limit(x, gemm1_alpha, gemm1_limit):
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


# ── MoE layer ──────────────────────────────────────────────────────────────
class MoELayer(torch.nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, dtype, device):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.w13_weight = torch.nn.Parameter(
            torch.randn(
                num_experts, 2 * intermediate_size, hidden_size,
                dtype=dtype, device=device,
            ) * 0.01,
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randn(
                num_experts, hidden_size, intermediate_size,
                dtype=dtype, device=device,
            ) * 0.01,
            requires_grad=False,
        )


# ── MoE forward (eager, no compile) ────────────────────────────────────────
def moe_forward(
    hidden_states, topk_weights, topk_ids,
    w13_weight, w2_weight, num_experts,
):
    """Native grouped_mm v2 MoE -- returns (num_tokens, hidden_size)."""
    device = hidden_states.device
    num_tokens, hidden_size = hidden_states.shape
    num_top_k = topk_ids.size(-1)

    expert_ids = topk_ids.reshape(-1)
    sample_weights = topk_weights.reshape(-1)
    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
    )
    current_hidden_states = hidden_states[token_idx]

    perm = torch.argsort(expert_ids, stable=True)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device, dtype=perm.dtype)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    current_states_g = current_hidden_states[perm]

    boundaries = torch.arange(
        1, num_experts + 1, device=device, dtype=expert_ids_g.dtype,
    )
    offsets = torch.searchsorted(expert_ids_g, boundaries).to(torch.int32)

    # GEMM1: (tokens_sorted, hidden) x (experts, hidden, 2*inter)^T
    w13_weight_t = w13_weight.transpose(-1, -2)
    gate_up_out = torch._grouped_mm(current_states_g, w13_weight_t, offsets)

    # SwiGLU activation
    hidden_act = swiglu_with_alpha_and_limit(gate_up_out, GEMM1_ALPHA, SWIGLU_LIMIT)
    hidden_act = hidden_act.to(current_states_g.dtype)

    # GEMM2: (tokens_sorted, inter) x (experts, inter, hidden)^T
    w2_weight_t = w2_weight.transpose(-1, -2)
    out_g = torch._grouped_mm(hidden_act, w2_weight_t, offsets)

    # Weight, unpermute, reduce across top-k
    out_g = out_g * sample_weights_g.unsqueeze(-1)
    out = out_g[inv_perm]
    return out.view(num_tokens, num_top_k, hidden_size).sum(dim=1).to(
        current_states_g.dtype
    )


# ── Synthetic routing ──────────────────────────────────────────────────────
def generate_topk_routing(num_tokens, num_experts, top_k, device):
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    return topk_weights.to(DTYPE), topk_ids.to(torch.int32)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)
    device = torch.device("cuda", rank)

    num_tokens = 4
    eps = 1e-5
    verbose = os.environ.get("TORCH_LOGS", "") != ""

    # ── Setup ──
    layer = MoELayer(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, DTYPE, device)
    hidden_states = torch.randn(
        num_tokens, HIDDEN_SIZE, device=device, dtype=DTYPE,
    )
    residual = torch.randn(
        num_tokens, HIDDEN_SIZE, device=device, dtype=DTYPE,
    )
    norm_weight = torch.ones(HIDDEN_SIZE, device=device, dtype=torch.float32)
    topk_weights, topk_ids = generate_topk_routing(
        num_tokens, NUM_EXPERTS, TOP_K, device,
    )

    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)
    symm_mem.get_symm_mem_workspace(
        group_name,
        min_size=num_tokens * HIDDEN_SIZE * torch.tensor([], dtype=DTYPE).element_size(),
    )

    # ── 1. Eager reference ──
    with torch.inference_mode():
        moe_out = moe_forward(
            hidden_states, topk_weights, topk_ids,
            layer.w13_weight, layer.w2_weight, NUM_EXPERTS,
        )
        ref_reduced = torch.ops._c10d_functional.all_reduce(
            moe_out, "sum", group_name,
        )
        ref_reduced = torch.ops._c10d_functional.wait_tensor(ref_reduced)
        ref_pre_norm = ref_reduced + residual
        ref_normed = F.rms_norm(ref_pre_norm, norm_weight.shape, norm_weight, eps)

    if rank == 0:
        print(f"Eager ref: normed norm={ref_normed.float().norm():.4f}")

    # ── 2. Compiled + fused Lamport ──
    # The compiled function takes the MoE output and does
    # allreduce → residual add → rmsnorm in one fused kernel.
    @torch.compile(options={
        "_fuse_symm_mem_comms": True,
        "_symm_mem_sync_mode": "lamport",
    })
    def fused_ar_norm(moe_output, res, w, gn):
        reduced = all_reduce(moe_output, "sum", group=gn)
        h = reduced + res
        normed = F.rms_norm(h, w.shape, w, eps)
        return normed, h

    with torch.inference_mode():
        (normed, pre_norm), code = run_and_get_code(
            fused_ar_norm, moe_out, residual, norm_weight, group_name,
        )

    if verbose and rank == 0:
        print("\n=== Generated code (allreduce + norm kernel) ===")
        for c in code:
            print(c)

    torch.testing.assert_close(normed, ref_normed, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(pre_norm, ref_pre_norm, atol=2e-2, rtol=2e-2)

    joined = "\n".join(code)
    assert "_lamport_poll_all_peers" in joined, "Missing Lamport poll"
    assert "_lamport_clear_old_slot" in joined, "Missing Lamport epilogue"

    if rank == 0:
        print(f"Fused:     normed norm={normed.float().norm():.4f}")
        print("PASSED: eager vs compiled+fused match")

    # ── 3. Compiled end-to-end: MoE + allreduce + norm in one compile ──
    # This is the ambitious case: the entire MoE → allreduce → norm
    # pipeline in a single compiled graph.  The MoE output (pointwise
    # tail: weight * unpermute → view+sum) is the upstream op that
    # should fuse into the allreduce kernel.
    @torch.compile(options={
        "_fuse_symm_mem_comms": True,
        "_symm_mem_sync_mode": "lamport",
    })
    def e2e_moe_ar_norm(
        hs, tw, ti, w13, w2, res, w, gn,
    ):
        moe = moe_forward(hs, tw, ti, w13, w2, NUM_EXPERTS)
        reduced = all_reduce(moe, "sum", group=gn)
        h = reduced + res
        normed = F.rms_norm(h, w.shape, w, eps)
        return normed, h

    with torch.inference_mode():
        (normed_e2e, pre_norm_e2e), code_e2e = run_and_get_code(
            e2e_moe_ar_norm,
            hidden_states, topk_weights, topk_ids,
            layer.w13_weight, layer.w2_weight,
            residual, norm_weight, group_name,
        )

    if verbose and rank == 0:
        print("\n=== Generated code (e2e MoE + allreduce + norm) ===")
        for c in code_e2e:
            print(c)

    torch.testing.assert_close(normed_e2e, ref_normed, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(pre_norm_e2e, ref_pre_norm, atol=2e-2, rtol=2e-2)

    if rank == 0:
        print(f"E2E:       normed norm={normed_e2e.float().norm():.4f}")
        print("PASSED: e2e compiled MoE + allreduce + norm matches eager")

    # ── 4. Quick latency comparison ──
    if rank == 0:
        print("\n--- Latency (10 iters, rank 0) ---")

    torch.cuda.synchronize()
    warmup = 5
    iters = 10

    # Eager
    for _ in range(warmup):
        with torch.inference_mode():
            m = moe_forward(
                hidden_states, topk_weights, topk_ids,
                layer.w13_weight, layer.w2_weight, NUM_EXPERTS,
            )
            r = torch.ops._c10d_functional.all_reduce(m, "sum", group_name)
            r = torch.ops._c10d_functional.wait_tensor(r)
            _ = F.rms_norm(r + residual, norm_weight.shape, norm_weight, eps)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.inference_mode():
            m = moe_forward(
                hidden_states, topk_weights, topk_ids,
                layer.w13_weight, layer.w2_weight, NUM_EXPERTS,
            )
            r = torch.ops._c10d_functional.all_reduce(m, "sum", group_name)
            r = torch.ops._c10d_functional.wait_tensor(r)
            _ = F.rms_norm(r + residual, norm_weight.shape, norm_weight, eps)
    torch.cuda.synchronize()
    eager_us = (time.perf_counter() - t0) / iters * 1e6

    # Compiled e2e
    for _ in range(warmup):
        with torch.inference_mode():
            e2e_moe_ar_norm(
                hidden_states, topk_weights, topk_ids,
                layer.w13_weight, layer.w2_weight,
                residual, norm_weight, group_name,
            )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.inference_mode():
            e2e_moe_ar_norm(
                hidden_states, topk_weights, topk_ids,
                layer.w13_weight, layer.w2_weight,
                residual, norm_weight, group_name,
            )
    torch.cuda.synchronize()
    compiled_us = (time.perf_counter() - t0) / iters * 1e6

    if rank == 0:
        print(f"  Eager MoE + allreduce + norm:    {eager_us:8.1f} μs")
        print(f"  Compiled e2e (Lamport fused):     {compiled_us:8.1f} μs")

    dist.destroy_process_group()


if __name__ == "__main__":
    timeout = int(os.environ.get("TEST_TIMEOUT", "120"))

    def _timeout_handler(signum, frame):
        print(f"TIMEOUT after {timeout}s — likely a hang in CUDA graph replay", file=sys.stderr)
        sys.exit(1)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    main()
