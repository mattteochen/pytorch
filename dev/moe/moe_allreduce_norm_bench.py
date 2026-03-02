"""
Benchmark: MoE → all_reduce → residual add → RMSNorm  (end-to-end TP pattern).

Three variants compared under CUDA graphs:

  1. triton_flashinfer  – SGLang Triton fused_experts + FlashInfer allreduce+norm
  2. compiled_p2p       – Native grouped_mm v2 compiled with inductor P2P
                          allreduce+norm fusion (_fuse_symm_mem_comms)
  3. eager_baseline     – Native grouped_mm v2 + NCCL all_reduce + F.rms_norm (eager)

All variants use the same gpt-oss-20b MoE config as dev/moe/moe.py and identical
synthetic routing so outputs are directly comparable.  Routing (argsort,
searchsorted) is included inside each benchmarked function, matching how
fused_experts and run_native_grouped_mm_v2 work in production.

Launch:
    torchrun --nproc_per_node=NUM_GPUS dev/moe_allreduce_norm_bench.py
    torchrun --nproc_per_node=NUM_GPUS dev/moe_allreduce_norm_bench.py --num-tokens 1 32
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F


# ── gpt-oss-20b model constants (must match dev/moe/moe.py) ─────────────────
NUM_EXPERTS = 32
TOP_K = 4
HIDDEN_SIZE = 2880
INTERMEDIATE_SIZE = 2880
GEMM1_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
DTYPE = torch.bfloat16
EPS = 1e-5

torch._inductor.config.triton.cudagraphs = False


# ── Activation (uncompiled so it can be inlined by the outer torch.compile) ──
def _swiglu(x, alpha, limit):
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1)


# ── Mock MoE weights ─────────────────────────────────────────────────────────
class MoEWeights:
    def __init__(self, num_experts, hidden, inter, dtype, device):
        self.num_experts = num_experts
        self.w13 = torch.randn(
            num_experts, 2 * inter, hidden, dtype=dtype, device=device
        ) * 0.01
        self.w2 = torch.randn(
            num_experts, hidden, inter, dtype=dtype, device=device
        ) * 0.01
        self.b13 = torch.randn(
            num_experts, 2 * inter, dtype=dtype, device=device
        ) * 0.01
        self.b2 = torch.randn(
            num_experts, hidden, dtype=dtype, device=device
        ) * 0.01
        self.norm_weight = torch.ones(hidden, dtype=torch.float32, device=device)


# ── Synthetic routing (deterministic for correctness comparison) ─────────────
def generate_routing(num_tokens, num_experts, top_k, device):
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1).to(DTYPE)
    topk_ids = topk_ids.to(torch.int32)
    return topk_weights, topk_ids, logits


# ── Full MoE + allreduce + norm (routing included, matches moe.py pattern) ──
def _moe_ar_norm(
    hidden_states, topk_weights, topk_ids,
    w13, b13, w2, b2, num_experts,
    residual, norm_weight, group_name, eps,
):
    """Routing + grouped_mm v2 + allreduce + residual add + RMSNorm."""
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
    current_hidden = hidden_states[token_idx]

    perm = torch.argsort(expert_ids, stable=True)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device, dtype=perm.dtype)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    current_states_g = current_hidden[perm]

    boundaries = torch.arange(
        1, num_experts + 1, device=device, dtype=expert_ids_g.dtype
    )
    offsets = torch.searchsorted(expert_ids_g, boundaries).to(torch.int32)

    # grouped_mm v2 (view+sum)
    gate_up = torch._grouped_mm(current_states_g, w13.transpose(-1, -2), offsets)
    gate_up = gate_up + b13[expert_ids_g]
    activated = _swiglu(gate_up, GEMM1_ALPHA, SWIGLU_LIMIT)
    activated = activated.to(current_states_g.dtype)

    out_g = torch._grouped_mm(activated, w2.transpose(-1, -2), offsets)
    out_g = out_g + b2[expert_ids_g]
    out_g = out_g * sample_weights_g.unsqueeze(-1)
    out = out_g[inv_perm]
    moe_out = out.view(num_tokens, num_top_k, hidden_size).sum(dim=1).to(
        current_states_g.dtype
    )

    # allreduce + residual + norm
    reduced = funcol.all_reduce(moe_out, "sum", group_name)
    h = reduced + residual
    normed = F.rms_norm(h, norm_weight.shape, norm_weight, eps)
    return normed, h, moe_out


# ── Variant 2: compiled P2P (full pipeline in one compile region) ────────────
def _make_compiled_p2p(eps):
    @torch.compile(options={
        "_fuse_symm_mem_comms": True,
        "_symm_mem_sync_mode": "lamport",
        "combo_kernels": True,
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "TRITON",
    })
    def fn(
        hidden_states, topk_weights, topk_ids,
        w13, b13, w2, b2, num_experts,
        residual, norm_weight, group_name,
    ):
        return _moe_ar_norm(
            hidden_states, topk_weights, topk_ids,
            w13, b13, w2, b2, num_experts,
            residual, norm_weight, group_name, eps,
        )

    return fn


# ── CUDA graph helpers (from dev/moe/moe.py) ────────────────────────────────
def capture_cuda_graph(fn, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_output = fn()
    torch.cuda.synchronize()
    return graph, static_output


def bench_cuda_graph(graph, warmup, iters, label):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_us = elapsed / iters * 1e6
    return avg_us


# ── Variant 1: Triton fused_experts + FlashInfer ────────────────────────────
def _try_setup_triton_flashinfer(rank, world_size, device, max_tokens):
    """Returns (make_call_fn, cleanup_fn) or (None, error_str)."""
    try:
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
        try:
            from sglang.srt.server_args import get_global_server_args
            get_global_server_args()
        except ValueError:
            server_args = ServerArgs(model_path="dummy")
            set_global_server_args_for_scheduler(server_args)

        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            fused_experts_impl,
        )
        import flashinfer.comm as flashinfer_comm
    except ImportError as e:
        return None, str(e)

    fi_ipc_handles, fi_workspace = (
        flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            tp_rank=rank,
            tp_size=world_size,
            max_token_num=max(max_tokens, 4096),
            hidden_dim=HIDDEN_SIZE,
            group=dist.group.WORLD,
        )
    )

    def make_call(layer, hidden_states, topk_weights, topk_ids, residual):
        norm_out = torch.empty_like(hidden_states)
        residual_out = torch.empty_like(hidden_states)
        moe_out_saved = [None]

        def call():
            moe_out = fused_experts_impl(
                hidden_states=hidden_states,
                w1=layer.w13, w2=layer.w2,
                topk_weights=topk_weights, topk_ids=topk_ids,
                b1=layer.b13, b2=layer.b2,
                inplace=False, activation="silu", is_gated=True,
                apply_router_weight_on_input=False,
                gemm1_alpha=GEMM1_ALPHA, gemm1_limit=SWIGLU_LIMIT,
            )
            moe_out_saved[0] = moe_out.clone()
            flashinfer_comm.trtllm_allreduce_fusion(
                allreduce_in=moe_out,
                token_num=moe_out.shape[0],
                residual_in=residual,
                residual_out=residual_out,
                norm_out=norm_out,
                rms_gamma=layer.norm_weight,
                rms_eps=EPS,
                hidden_dim=HIDDEN_SIZE,
                workspace_ptrs=fi_workspace,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                allreduce_out=None,
                quant_out=None, scale_out=None,
                layout_code=None, scale_factor=None,
                use_oneshot=None,
                world_rank=rank,
                world_size=world_size,
                launch_with_pdl=True,
                trigger_completion_at_end=True,
                fp32_acc=True,
            )
            return norm_out, residual_out, moe_out_saved[0]

        return call

    def cleanup():
        flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(
            fi_ipc_handles, dist.group.WORLD
        )

    return (make_call, cleanup), None


def _warmup(fn, warmup=10):
    """Run warmup iterations and return the last output (eager, no CUDA graph)."""
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    return out


def _bench(fn, warmup, iters, no_cuda_graph):
    """Benchmark only (assumes warmup already done). Returns avg_us."""
    if no_cuda_graph:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6

    graph, _ = capture_cuda_graph(fn, warmup=3)
    return bench_cuda_graph(graph, warmup, iters, "")


def _check(normed, ref_normed):
    if torch.allclose(normed, ref_normed, atol=1e-2, rtol=1e-2):
        return "PASS"
    diff = (normed.float() - ref_normed.float()).abs()
    return f"FAIL (max={diff.max().item():.4f}, mean={diff.mean().item():.4f})"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Bench MoE → allreduce → norm: 3 variants"
    )
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[1])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args, _ = parser.parse_known_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    group_name = dist.group.WORLD.group_name

    symm_mem.enable_symm_mem_for_group(group_name)
    max_tokens = max(args.num_tokens)
    symm_mem.get_symm_mem_workspace(group_name, min_size=max_tokens * HIDDEN_SIZE * 2)

    if rank == 0:
        print(f"Config: gpt-oss-20b | {NUM_EXPERTS} experts | top_k={TOP_K} | "
              f"hidden={HIDDEN_SIZE} | intermediate={INTERMEDIATE_SIZE}")
        print(f"Tokens: {args.num_tokens} | dtype: {DTYPE} | world_size: {world_size}")
        print(f"CUDA graphs: {'disabled' if args.no_cuda_graph else 'enabled'}\n")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    layer = MoEWeights(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, DTYPE, device)

    fi_setup, fi_err = _try_setup_triton_flashinfer(rank, world_size, device, max_tokens)

    compiled_fn = _make_compiled_p2p(EPS)

    all_results: dict[int, list[tuple[str, float, str]]] = {}

    for num_tokens in args.num_tokens:
        dist.barrier()
        torch.cuda.synchronize()
        torch._dynamo.reset()

        from torch._inductor.runtime.symm_mem_helpers import _symm_mem_cache
        _symm_mem_cache.clear()

        if rank == 0:
            print(f"\n{'#' * 70}")
            print(f"  NUM_TOKENS = {num_tokens}")
            print(f"{'#' * 70}")

        hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, dtype=DTYPE, device=device)
        residual = torch.randn_like(hidden_states)
        topk_weights, topk_ids, _logits = generate_routing(
            num_tokens, NUM_EXPERTS, TOP_K, device
        )

        # Clone all inputs/params so each variant gets independent copies
        # (guards against in-place mutation by kernels or CUDA graph capture).
        def _clone_inputs():
            return (
                hidden_states.clone(),
                topk_weights.clone(),
                topk_ids.clone(),
                layer.w13.clone(),
                layer.b13.clone(),
                layer.w2.clone(),
                layer.b2.clone(),
                residual.clone(),
                layer.norm_weight.clone(),
            )

        results: list[tuple[str, float, str]] = []

        # ── Eager baseline (NCCL, no P2P — correctness reference) ──────
        e_hs, e_tw, e_ti, e_w13, e_b13, e_w2, e_b2, e_res, e_nw = _clone_inputs()

        def eager_call():
            return _moe_ar_norm(
                e_hs, e_tw, e_ti,
                e_w13, e_b13, e_w2, e_b2, NUM_EXPERTS,
                e_res, e_nw, group_name, EPS,
            )

        eager_out = _warmup(eager_call, warmup=5)
        eager_normed = eager_out[0].clone()
        eager_h = eager_out[1].clone()
        eager_moe = eager_out[2].clone()

        # ── Build variant closures with cloned inputs ───────────────────
        c_hs, c_tw, c_ti, c_w13, c_b13, c_w2, c_b2, c_res, c_nw = _clone_inputs()

        def compiled_call():
            return compiled_fn(
                c_hs, c_tw, c_ti,
                c_w13, c_b13, c_w2, c_b2, NUM_EXPERTS,
                c_res, c_nw, group_name,
            )

        fi_call = None
        if fi_setup is not None:
            make_call, _cleanup = fi_setup
            f_hs, f_tw, f_ti, _, _, _, _, f_res, _ = _clone_inputs()
            fi_call = make_call(layer, f_hs, f_tw, f_ti, f_res)

        # ── Warmup all variants (eager, before CUDA graphs) ────────────
        compiled_out = _warmup(compiled_call, warmup=args.warmup)
        compiled_normed = compiled_out[0].clone()
        compiled_h = compiled_out[1].clone() if len(compiled_out) > 1 else None
        compiled_moe = compiled_out[2].clone() if len(compiled_out) > 2 else None

        fi_normed = None
        fi_h = None
        fi_moe = None
        if fi_call is not None:
            try:
                fi_out = _warmup(fi_call, warmup=args.warmup)
                fi_normed = fi_out[0].clone()
                fi_h = fi_out[1].clone() if len(fi_out) > 1 else None
                fi_moe = fi_out[2].clone() if len(fi_out) > 2 else None
            except Exception as e:
                if rank == 0:
                    print(f"  triton_flashinfer warmup: ERROR - {e}")
                fi_call = None

        # ── Correctness comparison (on eager warmup outputs) ───────────
        if rank == 0:
            def _stats(a, b):
                d = (a.float() - b.float()).abs()
                return d.max().item(), d.mean().item()

            def _range(t):
                return f"[{t.min().item():.4f}, {t.max().item():.4f}]"

            print(f"\n  Correctness (eager warmup, before CUDA graphs):")
            print(f"  {'':30s} {'norm':>10s} {'out range':>24s}")
            print(f"  {'eager_baseline (REF)':30s} {eager_normed.float().norm():10.4f} {_range(eager_normed):>24s}")
            print(f"  {'compiled_p2p':30s} {compiled_normed.float().norm():10.4f} {_range(compiled_normed):>24s}")
            if fi_normed is not None:
                print(f"  {'triton_flashinfer':30s} {fi_normed.float().norm():10.4f} {_range(fi_normed):>24s}")

            print(f"\n  {'Diff vs eager_baseline':30s} {'moe_out max':>14s} {'h max':>14s} {'normed max':>14s}")
            print(f"  {'-' * 30} {'-' * 14} {'-' * 14} {'-' * 14}")

            mx_m = _stats(compiled_moe, eager_moe)[0] if compiled_moe is not None else float('nan')
            mx_h = _stats(compiled_h, eager_h)[0] if compiled_h is not None else float('nan')
            mx_n = _stats(compiled_normed, eager_normed)[0]
            print(f"  {'compiled_p2p':30s} {mx_m:14.6e} {mx_h:14.6e} {mx_n:14.6e}")

            if fi_normed is not None:
                mx_m = _stats(fi_moe, eager_moe)[0] if fi_moe is not None else float('nan')
                mx_h = _stats(fi_h, eager_h)[0] if fi_h is not None else float('nan')
                mx_n = _stats(fi_normed, eager_normed)[0]
                print(f"  {'triton_flashinfer':30s} {mx_m:14.6e} {mx_h:14.6e} {mx_n:14.6e}")

        # ── Speed benchmark ────────────────────────────────────────────
        compiled_us = _bench(compiled_call, args.warmup, args.iters, args.no_cuda_graph)
        results.append(("compiled_p2p", compiled_us, "REF"))

        if fi_call is not None:
            try:
                fi_us = _bench(fi_call, args.warmup, args.iters, args.no_cuda_graph)
                fi_ok = _check(fi_normed, compiled_normed)
                results.append(("triton_flashinfer", fi_us, fi_ok))
            except Exception as e:
                if rank == 0:
                    print(f"  triton_flashinfer bench: ERROR - {e}")
        elif rank == 0:
            print(f"  triton_flashinfer: SKIPPED ({fi_err})")

        all_results[num_tokens] = results

        # ── Print results ───────────────────────────────────────────────
        if rank == 0:
            baseline_us = results[0][1]
            print(f"\n  {'Variant':<30s} {'us/iter':>10s} {'vs baseline':>12s}  {'correctness'}")
            print(f"  {'-' * 30} {'-' * 10} {'-' * 12}  {'-' * 20}")
            for name, us, ok in results:
                speedup = f"{baseline_us / us:.2f}x" if us > 0 else "--"
                print(f"  {name:<30s} {us:10.1f} {speedup:>12s}  {ok}")

    # ── Combined table ────────────────────────────────────────────────
    if rank == 0 and len(args.num_tokens) > 1:
        variant_names = []
        for tc in args.num_tokens:
            for name, _, _ in all_results.get(tc, []):
                if name not in variant_names:
                    variant_names.append(name)

        print(f"\n{'=' * 70}")
        print(f"  COMBINED  (hidden={HIDDEN_SIZE}, world_size={world_size})")
        print(f"{'=' * 70}")
        header = f"  {'Variant':<30s}"
        for tc in args.num_tokens:
            header += f" {'T=' + str(tc):>10s}"
        print(header)
        print(f"  {'-' * 30}" + f" {'-' * 10}" * len(args.num_tokens))
        for vname in variant_names:
            row = f"  {vname:<30s}"
            for tc in args.num_tokens:
                us = next(
                    (u for n, u, _ in all_results.get(tc, []) if n == vname), None
                )
                row += f" {us:10.1f}" if us is not None else f" {'--':>10s}"
            print(row)
        print()

    if fi_setup is not None:
        _, cleanup = fi_setup
        cleanup()

    dist.destroy_process_group()
    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    main()
