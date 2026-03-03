"""
Benchmark: MoE → allreduce → RMSNorm, two implementations.

  Path A (ours): native grouped_mm v2 → Lamport fused allreduce+norm
                 (single torch.compile graph)
  Path B (baseline): SGLang fused_triton MoE → FlashInfer trtllm_allreduce_fusion

Both paths use CUDA graphs for latency measurement.
Correctness is checked during eager warmup against an eager reference.

Usage (requires 2+ GPUs with P2P access):
  torchrun --nproc_per_node=2 dev/moe/bench_moe_ar_norm.py
  torchrun --nproc_per_node=2 dev/moe/bench_moe_ar_norm.py --num-tokens 4
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce

torch._inductor.config.triton.cudagraphs = False

# ── gpt-oss-20b model constants ────────────────────────────────────────────
NUM_EXPERTS = 32
TOP_K = 4
HIDDEN_SIZE = 2880
INTERMEDIATE_SIZE = 2880
GEMM1_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
DTYPE = torch.bfloat16
EPS = 1e-5


# ── Activation ──────────────────────────────────────────────────────────────
def swiglu_with_alpha_and_limit(x, gemm1_alpha, gemm1_limit):
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


# ── MoE forward (eager, no compile) ────────────────────────────────────────
def moe_forward(
    hidden_states, topk_weights, topk_ids,
    w13_weight, w13_weight_bias, w2_weight, w2_weight_bias, num_experts,
):
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

    w13_weight_t = w13_weight.transpose(-1, -2)
    gate_up_out = torch._grouped_mm(current_states_g, w13_weight_t, offsets)

    if w13_weight_bias is not None:
        gate_up_out = gate_up_out + w13_weight_bias[expert_ids_g]

    hidden_act = swiglu_with_alpha_and_limit(gate_up_out, GEMM1_ALPHA, SWIGLU_LIMIT)
    hidden_act = hidden_act.to(current_states_g.dtype)

    w2_weight_t = w2_weight.transpose(-1, -2)
    out_g = torch._grouped_mm(hidden_act, w2_weight_t, offsets)

    if w2_weight_bias is not None:
        out_g = out_g + w2_weight_bias[expert_ids_g]

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


# ── Layer (shared weights for both paths) ──────────────────────────────────
class MoELayer(torch.nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, dtype, device):
        super().__init__()
        self.num_experts = num_experts
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
        # SGLang fused_triton expects bias parameters
        self.w13_weight_bias = torch.nn.Parameter(
            torch.randn(
                num_experts, 2 * intermediate_size,
                dtype=dtype, device=device,
            ) * 0.01,
            requires_grad=False,
        )
        self.w2_weight_bias = torch.nn.Parameter(
            torch.randn(
                num_experts, hidden_size,
                dtype=dtype, device=device,
            ) * 0.01,
            requires_grad=False,
        )


# ── Benchmark ──────────────────────────────────────────────────────────────
def benchmark(name, fn, rank, warmup=10, iters=100, cuda_graph=True, cuda_graph_batch=10):
    """Measure latency. Uses CUDA graph replay when cuda_graph=True.

    Returns (avg_us, graph) where graph is a single-iteration CUDAGraph
    (for NVTX profiling), or None when cuda_graph=False.
    """
    if rank == 0:
        print(f"Benchmarking {name}...", flush=True)
    dist.barrier()

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    if cuda_graph:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()

        g_single = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_single, stream=stream):
            fn()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for _ in range(cuda_graph_batch):
                fn()
        torch.cuda.synchronize()
        dist.barrier()

        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters // cuda_graph_batch):
            g.replay()
        torch.cuda.synchronize()
        avg_us = (t0 - time.perf_counter()) / -iters * 1e6
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        avg_us = (t0 - time.perf_counter()) / -iters * 1e6
        g_single = None

    if rank == 0:
        print(f"  {name:40s} {avg_us:8.1f} μs/iter", flush=True)
    return avg_us, g_single


def check_correctness(name, result, ref, rank, atol=2e-2, rtol=2e-2):
    try:
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)
        if rank == 0:
            print(f"  {name}: PASS", flush=True)
        return True
    except AssertionError as e:
        if rank == 0:
            print(f"  {name}: FAIL — {e}", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cuda-graph-batch", type=int, default=10)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)
    device = torch.device("cuda", rank)

    num_tokens = args.num_tokens
    group_name = dist.group.WORLD.group_name

    # ── Setup ──
    layer = MoELayer(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, DTYPE, device)
    hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=DTYPE)
    residual = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=DTYPE)
    norm_weight = torch.ones(HIDDEN_SIZE, device=device, dtype=torch.float32)
    topk_weights, topk_ids = generate_topk_routing(
        num_tokens, NUM_EXPERTS, TOP_K, device,
    )

    symm_mem.enable_symm_mem_for_group(group_name)
    symm_mem.get_symm_mem_workspace(
        group_name,
        min_size=num_tokens * HIDDEN_SIZE * DTYPE.itemsize,
    )

    if rank == 0:
        print(f"Config: {num_tokens} tokens, {HIDDEN_SIZE} hidden, "
              f"{NUM_EXPERTS} experts, top-{TOP_K}, {world_size} GPUs",
              flush=True)

    # ── Eager reference ──
    if rank == 0:
        print("\n--- Eager reference ---", flush=True)
    with torch.inference_mode():
        moe_out_ref = moe_forward(
            hidden_states, topk_weights, topk_ids,
            layer.w13_weight, layer.w13_weight_bias,
            layer.w2_weight, layer.w2_weight_bias, NUM_EXPERTS,
        )
        ref_reduced = torch.ops._c10d_functional.all_reduce(
            moe_out_ref, "sum", group_name,
        )
        ref_reduced = torch.ops._c10d_functional.wait_tensor(ref_reduced)
        ref_pre_norm = ref_reduced + residual
        ref_normed = F.rms_norm(ref_pre_norm, norm_weight.shape, norm_weight, EPS)
    if rank == 0:
        print(f"  normed norm={ref_normed.float().norm():.4f}", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # Path A: E2E compiled (native grouped_mm v2 + Lamport fused AR+norm)
    # ══════════════════════════════════════════════════════════════════════
    if rank == 0:
        print("\n--- Path A: E2E compiled (MoE v2 + Lamport fused) ---", flush=True)

    @torch.compile(options={
        "combo_kernels": True,
        "_fuse_symm_mem_comms": True,
        "_symm_mem_sync_mode": "lamport",
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "TRITON"
    })
    def e2e_compiled(hs, tw, ti, w13, w13b, w2, w2b, res, w, gn):
        moe = moe_forward(hs, tw, ti, w13, w13b, w2, w2b, NUM_EXPERTS)
        reduced = all_reduce(moe, "sum", group=gn)
        h = reduced + res
        normed = F.rms_norm(h, w.shape, w, EPS)
        return normed, h

    def path_a_call():
        return e2e_compiled(
            hidden_states, topk_weights, topk_ids,
            layer.w13_weight, layer.w13_weight_bias,
            layer.w2_weight, layer.w2_weight_bias,
            residual, norm_weight, group_name,
        )

    # Correctness (eager warmup)
    with torch.inference_mode():
        normed_a, pre_norm_a = path_a_call()
    check_correctness("Path A normed", normed_a, ref_normed, rank)
    check_correctness("Path A pre_norm", pre_norm_a, ref_pre_norm, rank)

    # ══════════════════════════════════════════════════════════════════════
    # Path B: SGLang fused_triton MoE + FlashInfer AR+norm
    # ══════════════════════════════════════════════════════════════════════
    path_b_ok = False
    try:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl
        from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
        import flashinfer.comm as flashinfer_comm

        if rank == 0:
            print("\n--- Path B: SGLang fused_triton + FlashInfer AR+norm ---")

        # Init server args (needed by Triton config lookup)
        try:
            from sglang.srt.server_args import get_global_server_args
            get_global_server_args()
        except ValueError:
            server_args = ServerArgs(model_path="dummy")
            set_global_server_args_for_scheduler(server_args)

        # FlashInfer workspace
        fi_max_tokens = max(num_tokens, 4096)
        fi_ipc_handles, fi_workspace = (
            flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                tp_rank=rank,
                tp_size=world_size,
                max_token_num=fi_max_tokens,
                hidden_dim=HIDDEN_SIZE,
                group=dist.group.WORLD,
            )
        )

        fi_norm_out = torch.empty_like(hidden_states)
        fi_residual_out = torch.empty_like(hidden_states)

        def path_b_call():
            # MoE via SGLang fused_triton
            moe_out = fused_experts_impl(
                hidden_states=hidden_states,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                b1=layer.w13_weight_bias,
                b2=layer.w2_weight_bias,
                inplace=False,
                activation="silu",
                is_gated=True,
                apply_router_weight_on_input=False,
                gemm1_alpha=GEMM1_ALPHA,
                gemm1_limit=SWIGLU_LIMIT,
            )
            # AR + residual + RMSNorm via FlashInfer
            flashinfer_comm.trtllm_allreduce_fusion(
                allreduce_in=moe_out,
                token_num=moe_out.shape[0],
                residual_in=residual,
                residual_out=fi_residual_out,
                norm_out=fi_norm_out,
                rms_gamma=norm_weight,
                rms_eps=EPS,
                hidden_dim=HIDDEN_SIZE,
                workspace_ptrs=fi_workspace,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                allreduce_out=None,
                quant_out=None,
                scale_out=None,
                layout_code=None,
                scale_factor=None,
                use_oneshot=None, # self tuned
                world_rank=rank,
                world_size=world_size,
                launch_with_pdl=True,
                trigger_completion_at_end=True,
                fp32_acc=True,
            )

        # Correctness (eager warmup — run a few times for NCCL/Triton init)
        with torch.inference_mode():
            for _ in range(3):
                path_b_call()
        check_correctness("Path B normed", fi_norm_out, ref_normed, rank)
        check_correctness("Path B pre_norm", fi_residual_out, ref_pre_norm, rank)
        path_b_ok = True

    except (ImportError, AttributeError, RuntimeError) as e:
        if rank == 0:
            print(f"\nSkipping Path B (SGLang/FlashInfer): {e}")

    # ══════════════════════════════════════════════════════════════════════
    # Latency comparison
    # ══════════════════════════════════════════════════════════════════════
    if rank == 0:
        print(f"\n--- Latency ({args.iters} iters) ---")

    use_cg = not args.no_cuda_graph
    b_graph = None
    with torch.inference_mode():
        if path_b_ok:
            b_us, b_graph = benchmark(
                "Path B: fused_triton + FlashInfer AR+norm",
                path_b_call, rank,
                warmup=args.warmup, iters=args.iters,
                cuda_graph=use_cg, cuda_graph_batch=args.cuda_graph_batch,
            )
        a_us, a_graph = benchmark(
            "Path A: compiled MoE v2 + Lamport AR+norm",
            path_a_call, rank,
            warmup=args.warmup, iters=args.iters,
            cuda_graph=use_cg, cuda_graph_batch=args.cuda_graph_batch,
        )

    if rank == 0 and path_b_ok:
        speedup = b_us / a_us if a_us > 0 else float("inf")
        print(f"\n  Speedup (A vs B): {speedup:.2f}x")

    # ── NVTX-marked single replay for nsys profiling ──
    if args.nvtx:
        if rank == 0:
            print("\nRunning NVTX-marked iterations (cuda-graphed, 1x each)...")

        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range("path_a_compiled_moe_v2_lamport"):
            a_graph.replay()
        torch.cuda.synchronize()
        if b_graph is not None:
            with torch.cuda.nvtx.range("path_b_fused_triton_flashinfer"):
                b_graph.replay()
            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        if rank == 0:
            print("Done. Use 'nsys profile ...' to capture the trace.")

    # Cleanup
    if path_b_ok:
        try:
            flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(
                fi_ipc_handles, dist.group.WORLD
            )
        except Exception:
            pass

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
