"""
Standalone repro comparing 3 MoE backends on gpt-oss-20b config:
  1. Triton fused_experts (fused_moe_kernel)
  2. Triton-kernels matmul_ogs (the AUTO default on GB200/SM100)
  3. Native grouped_mm (torch.compiled)

All paths use CUDA graphs for benchmarking.

Usage:
  python -m sglang.bench_moe_native
  python -m sglang.bench_moe_native --num-tokens 1 --warmup 5 --iters 100
  nsys profile -o moe_compare python -m sglang.bench_moe_native --num-tokens 1

Config from: https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json
"""

import argparse
import os
import time
from typing import Optional
import functools

import torch
import torch.nn.functional as F
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config

os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "TRITON")
# torch._inductor.config.max_autotune_gemm = True
# Manual CUDA graph capture is used below, so disable Inductor's internal
# cudagraph wrapping to avoid double-capture conflicts.
torch._inductor.config.triton.cudagraphs = False
# torch._inductor.config.combo_kernels = True

def compile_with_debug(fn, compile_kwargs=None, dynamo_kwargs=None, inductor_kwargs=None):
    """Wrap a function with torch.compile and enable debug output.

    Debug config is applied when compilation actually occurs (on first call).

    Args:
        fn: The function to compile.
        compile_kwargs: Dict of arguments passed to torch.compile (e.g. mode, fullgraph, dynamic).
        inductor_kwargs: Dict of extra inductor config overrides merged with debug defaults.
    """
    if compile_kwargs is None:
        compile_kwargs = {}
    if inductor_kwargs is None:
        inductor_kwargs = {}
    if dynamo_kwargs is None:
        dynamo_kwargs = {}

    if "mode" in compile_kwargs:
        mode_options = torch._inductor.list_mode_options(compile_kwargs.pop("mode"))
        inductor_kwargs = {**mode_options, **inductor_kwargs}

    compiled_fn = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal compiled_fn

        if compiled_fn is None:
            # First call - compile with debug settings and caching disabled
            with dynamo_config.patch(
                     verbose=True,
                     **dynamo_kwargs,
                 ), \
                 inductor_config.patch(
                    **inductor_kwargs, **{
                     "debug": True,
                     "trace.enabled": True,
                     "trace.fx_graph": True,
                     "trace.fx_graph_transformed": True,
                     "trace.output_code": True,
                     "fx_graph_cache": False,
                     "force_disable_caches": True
                 }):
                compiled_fn = torch.compile(fn, options=inductor_kwargs, **compile_kwargs)
                return compiled_fn(*args, **kwargs)

        # No recompilation flag. The function was already compiled with dynamo and inductor configs.
        # This early return avoids inductor context manager overhead.
        if "cache_size_limit" in dynamo_kwargs and dynamo_kwargs["cache_size_limit"] == 1:
            # We add error_on_recompile to avoid silent eager fallbacks
            with dynamo_config.patch(**dynamo_kwargs, error_on_recompile=True):
                return compiled_fn(*args, **kwargs)

        # Apply config on every call so recompilations
        # (triggered by dynamo guard failures) still use cpp_wrapper, capture_scalar_outputs, etc.
        with dynamo_config.patch(**dynamo_kwargs):
            #  inductor_config.patch(**inductor_kwargs):
            return compiled_fn(*args, **kwargs)

    return wrapper


# ── gpt-oss-20b model constants ──────────────────────────────────────────────
NUM_EXPERTS = 32
TOP_K = 4
HIDDEN_SIZE = 2880
INTERMEDIATE_SIZE = 2880  # gate+up fused → 2 * 2880 = 5760
GEMM1_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
DTYPE = torch.bfloat16


# ── Activation (torch.compiled, matches production) ──────────────────────────
@torch.compile
def swiglu_with_alpha_and_limit_compiled(x, gemm1_alpha, gemm1_limit):
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


# ── Minimal mock layer ───────────────────────────────────────────────────────
class MockFusedMoELayer(torch.nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, dtype, device):
        super().__init__()
        self.num_experts = num_experts
        self.w13_weight = torch.nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, hidden_size,
                        dtype=dtype, device=device) * 0.01,
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size,
                        dtype=dtype, device=device) * 0.01,
            requires_grad=False,
        )
        self.w13_weight_bias = torch.nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size,
                        dtype=dtype, device=device) * 0.01,
            requires_grad=False,
        )
        self.w2_weight_bias = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size,
                        dtype=dtype, device=device) * 0.01,
            requires_grad=False,
        )


# ── Server args init (needed by Triton config lookup) ────────────────────────
def _init_server_args_if_needed():
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    try:
        from sglang.srt.server_args import get_global_server_args
        get_global_server_args()
    except ValueError:
        server_args = ServerArgs(model_path="dummy")
        set_global_server_args_for_scheduler(server_args)


# ── 1. Triton fused_experts path ─────────────────────────────────────────────
def run_triton_fused_experts(layer, hidden_states, topk_weights, topk_ids):
    _init_server_args_if_needed()
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl

    return fused_experts_impl(
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


# ── 2. Triton-kernels matmul_ogs path (AUTO default on SM100) ────────────────
def run_triton_kernels(layer, hidden_states, router_logits):
    from triton_kernels.routing import routing

    from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
        triton_kernel_fused_experts_with_bias,
    )

    routing_data, gather_idx, scatter_idx = routing(
        router_logits, TOP_K, sm_first=False,
    )

    # matmul_ogs expects (E, K, N) layout; our weights are (E, N, K)
    w1_t = layer.w13_weight.transpose(-1, -2).contiguous()
    w2_t = layer.w2_weight.transpose(-1, -2).contiguous()
    # triton_kernels requires float32 biases to match accumulator dtype
    b1 = layer.w13_weight_bias.float()
    b2 = layer.w2_weight_bias.float()

    return triton_kernel_fused_experts_with_bias(
        hidden_states=hidden_states,
        w1=w1_t,
        w1_pcg=None,
        b1=b1,
        w2=w2_t,
        w2_pcg=None,
        b2=b2,
        routing_data=routing_data,
        gather_indx=gather_idx,
        scatter_indx=scatter_idx,
        inplace=False,
        activation="silu",
        apply_router_weight_on_input=False,
        gemm1_alpha=GEMM1_ALPHA,
        gemm1_clamp_limit=SWIGLU_LIMIT,
    )


# ── 3. Native grouped_mm path (torch.compiled) ──────────────────────────────
def _native_grouped_mm_core(
    current_states_g, expert_ids_g, sample_weights_g,
    w13_weight, w13_weight_bias, w2_weight, w2_weight_bias,
    offsets, inv_perm, token_idx, num_tokens, hidden_size,
):
    final_hidden_states = torch.zeros(
        num_tokens, hidden_size,
        dtype=current_states_g.dtype, device=current_states_g.device,
    )

    w13_weight_t = w13_weight.transpose(-1, -2)
    gate_up_out = torch._grouped_mm(current_states_g, w13_weight_t, offsets)

    if w13_weight_bias is not None:
        gate_up_out = gate_up_out + w13_weight_bias[expert_ids_g]

    hidden_after_activation = swiglu_with_alpha_and_limit_compiled(
        gate_up_out, GEMM1_ALPHA, SWIGLU_LIMIT
    )
    hidden_after_activation = hidden_after_activation.to(current_states_g.dtype)

    w2_weight_t = w2_weight.transpose(-1, -2)
    out_per_sample_g = torch._grouped_mm(hidden_after_activation, w2_weight_t, offsets)

    if w2_weight_bias is not None:
        out_per_sample_g = out_per_sample_g + w2_weight_bias[expert_ids_g]

    out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(-1)
    out_per_sample = out_per_sample_g[inv_perm]
    scatter_idx = token_idx.unsqueeze(-1).expand_as(out_per_sample)
    final_hidden_states.scatter_add_(0, scatter_idx, out_per_sample)

    return final_hidden_states.to(current_states_g.dtype)


def _native_grouped_mm_core_v2(
    current_states_g, expert_ids_g, sample_weights_g,
    w13_weight, w13_weight_bias, w2_weight, w2_weight_bias,
    offsets, inv_perm, num_tokens, num_top_k, hidden_size,
):
    w13_weight_t = w13_weight.transpose(-1, -2)
    gate_up_out = torch._grouped_mm(current_states_g, w13_weight_t, offsets)

    if w13_weight_bias is not None:
        gate_up_out = gate_up_out + w13_weight_bias[expert_ids_g]

    hidden_after_activation = swiglu_with_alpha_and_limit_compiled(
        gate_up_out, GEMM1_ALPHA, SWIGLU_LIMIT
    )
    hidden_after_activation = hidden_after_activation.to(current_states_g.dtype)

    w2_weight_t = w2_weight.transpose(-1, -2)
    out_per_sample_g = torch._grouped_mm(hidden_after_activation, w2_weight_t, offsets)

    if w2_weight_bias is not None:
        out_per_sample_g = out_per_sample_g + w2_weight_bias[expert_ids_g]

    out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(-1)
    out_per_sample = out_per_sample_g[inv_perm]
    return out_per_sample.view(num_tokens, num_top_k, hidden_size).sum(dim=1).to(current_states_g.dtype)


def run_native_grouped_mm_v2(layer, hidden_states, topk_weights, topk_ids):
    device = hidden_states.device
    num_tokens, hidden_size = hidden_states.shape
    num_top_k = topk_ids.size(-1)
    num_experts = layer.num_experts

    expert_ids = topk_ids.reshape(-1)
    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
    )

    sample_weights = topk_weights.reshape(-1)
    current_hidden_states = hidden_states[token_idx]

    perm = torch.argsort(expert_ids, stable=True)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device, dtype=perm.dtype)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    current_states_g = current_hidden_states[perm]

    boundaries = torch.arange(1, num_experts + 1, device=device, dtype=expert_ids_g.dtype)
    offsets = torch.searchsorted(expert_ids_g, boundaries).to(torch.int32)

    return _native_grouped_mm_core_v2(
        current_states_g, expert_ids_g, sample_weights_g,
        layer.w13_weight, layer.w13_weight_bias,
        layer.w2_weight, layer.w2_weight_bias,
        offsets, inv_perm, num_tokens, num_top_k, hidden_size,
    )

run_native_grouped_mm_v2 = torch.compile(
    run_native_grouped_mm_v2,
    options={"combo_kernels": True, "max_autotune_gemm": True},
)
run_native_grouped_mm_v2 = compile_with_debug(
    run_native_grouped_mm_v2,
    inductor_kwargs={"combo_kernels": True, "max_autotune_gemm": True},
)


def run_native_grouped_mm(layer, hidden_states, topk_weights, topk_ids):
    device = hidden_states.device
    num_tokens, hidden_size = hidden_states.shape
    num_top_k = topk_ids.size(-1)
    num_experts = layer.num_experts

    expert_ids = topk_ids.reshape(-1)
    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
    )

    sample_weights = topk_weights.reshape(-1)
    current_hidden_states = hidden_states[token_idx]

    perm = torch.argsort(expert_ids, stable=True)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device, dtype=perm.dtype)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    current_states_g = current_hidden_states[perm]

    boundaries = torch.arange(1, num_experts + 1, device=device, dtype=expert_ids_g.dtype)
    offsets = torch.searchsorted(expert_ids_g, boundaries).to(torch.int32)

    return _native_grouped_mm_core(
        current_states_g, expert_ids_g, sample_weights_g,
        layer.w13_weight, layer.w13_weight_bias,
        layer.w2_weight, layer.w2_weight_bias,
        offsets, inv_perm, token_idx, num_tokens, hidden_size,
    )

# run_native_grouped_mm = compile_with_debug(run_native_grouped_mm, inductor_kwargs={"combo_kernels": True, "max_autotune_gemm": True})
run_native_grouped_mm = torch.compile(run_native_grouped_mm, options={"combo_kernels": True, "max_autotune_gemm": True})


# ── Synthetic routing ────────────────────────────────────────────────────────
def generate_topk_routing(num_tokens, num_experts, top_k, device):
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    return topk_weights.to(DTYPE), topk_ids.to(torch.int32), logits


# ── CUDA graph capture + replay benchmark ────────────────────────────────────
def capture_cuda_graph(fn):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = fn()
    return graph, static_output


def bench_cuda_graph(graph, warmup, iters, label):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_us = elapsed / iters * 1e6
    print(f"  {label:40s}  {avg_us:8.1f} μs  ({iters} iters)")
    return avg_us


def bench_eager(fn, warmup, iters, label):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_us = elapsed / iters * 1e6
    print(f"  {label:40s}  {avg_us:8.1f} μs  ({iters} iters)")
    return out, avg_us


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Bench 3 MoE backends: Triton / triton-kernels / native grouped_mm"
    )
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    num_tokens = args.num_tokens

    print(f"Config: gpt-oss-20b | {NUM_EXPERTS} experts | top_k={TOP_K} | "
          f"hidden={HIDDEN_SIZE} | intermediate={INTERMEDIATE_SIZE}")
    print(f"Tokens: {num_tokens} | dtype: {DTYPE}")
    print(f"CUDA graphs: {'disabled' if args.no_cuda_graph else 'enabled'}\n")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    layer = MockFusedMoELayer(
        NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, DTYPE, device
    )

    hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, dtype=DTYPE, device=device)
    topk_weights, topk_ids, router_logits = generate_topk_routing(
        num_tokens, NUM_EXPERTS, TOP_K, device
    )

    triton_fn = lambda: run_triton_fused_experts(
        layer, hidden_states, topk_weights, topk_ids
    )
    tk_fn = lambda: run_triton_kernels(
        layer, hidden_states, router_logits
    )
    native_fn = lambda: run_native_grouped_mm(
        layer, hidden_states, topk_weights, topk_ids
    )
    native_v2_fn = lambda: run_native_grouped_mm_v2(
        layer, hidden_states, topk_weights, topk_ids
    )

    # ── Warmup all paths (torch.compile + triton autotune) ──
    print("Warming up torch.compile + Triton autotune...")
    for _ in range(5):
        triton_out = triton_fn()
        tk_out = tk_fn()
        native_out = native_fn()
        native_v2_out = native_v2_fn()
    torch.cuda.synchronize()

    # ── Correctness ──
    triton_f = triton_out.float()
    tk_f = tk_out.float()
    native_f = native_out.float()
    native_v2_f = native_v2_out.float()

    print(f"\nOutput comparison:")
    print(f"  {'Backend':40s}  {'norm':>10s}  {'max Δ vs Triton':>16s}")
    print(f"  {'Triton fused_experts':40s}  {triton_f.norm():10.4f}  {'(reference)':>16s}")
    print(f"  {'Triton-kernels matmul_ogs':40s}  {tk_f.norm():10.4f}"
          f"  {(tk_f - triton_f).abs().max().item():16.6e}")
    print(f"  {'Native grouped_mm (compiled)':40s}  {native_f.norm():10.4f}"
          f"  {(native_f - triton_f).abs().max().item():16.6e}")
    print(f"  {'Native grouped_mm v2 (view+sum)':40s}  {native_v2_f.norm():10.4f}"
          f"  {(native_v2_f - triton_f).abs().max().item():16.6e}")

    # ── Eager benchmark ──
    print(f"\nEager latency:")
    _, triton_eager = bench_eager(triton_fn, args.warmup, args.iters,
                                  "Triton fused_experts")
    _, tk_eager = bench_eager(tk_fn, args.warmup, args.iters,
                              "Triton-kernels matmul_ogs (AUTO/SM100)")
    _, native_eager = bench_eager(native_fn, args.warmup, args.iters,
                                  "Native grouped_mm (compiled)")
    _, native_v2_eager = bench_eager(native_v2_fn, args.warmup, args.iters,
                                     "Native grouped_mm v2 (view+sum)")

    if args.no_cuda_graph:
        _print_summary(triton_eager, tk_eager, native_eager, native_v2_eager, "Eager")
        return

    # ── CUDA graph benchmark ──
    print(f"\nCapturing CUDA graphs...")
    triton_graph, _ = capture_cuda_graph(triton_fn)
    tk_graph, _ = capture_cuda_graph(tk_fn)
    native_graph, _ = capture_cuda_graph(native_fn)
    native_v2_graph, _ = capture_cuda_graph(native_v2_fn)

    print(f"\nCUDA graph latency:")
    triton_graph_us = bench_cuda_graph(
        triton_graph, args.warmup, args.iters, "Triton fused_experts"
    )
    tk_graph_us = bench_cuda_graph(
        tk_graph, args.warmup, args.iters, "Triton-kernels matmul_ogs (AUTO/SM100)"
    )
    native_graph_us = bench_cuda_graph(
        native_graph, args.warmup, args.iters, "Native grouped_mm (compiled)"
    )
    native_v2_graph_us = bench_cuda_graph(
        native_v2_graph, args.warmup, args.iters, "Native grouped_mm v2 (view+sum)"
    )

    _print_summary(triton_eager, tk_eager, native_eager, native_v2_eager, "Eager")
    _print_summary(triton_graph_us, tk_graph_us, native_graph_us, native_v2_graph_us, "CUDA graph")

    # ── NVTX-marked iteration (CUDA-graphed) for nsys profiling ──
    print("\nRunning NVTX-marked iterations (cuda-graphed, 1x each)...")
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("triton_fused_experts"):
        triton_graph.replay()
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("native_grouped_mm"):
        native_graph.replay()
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("native_grouped_mm_v2"):
        native_v2_graph.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("Done. Use 'nsys profile ...' to capture the trace.")


def _print_summary(triton_us, tk_us, native_us, native_v2_us, mode):
    fastest = min(triton_us, tk_us, native_us, native_v2_us)
    print(f"\n  {mode} summary (lower is better):")
    print(f"    {'Triton fused_experts':40s}  {triton_us:8.1f} μs  ({triton_us/fastest:.2f}x)")
    print(f"    {'Triton-kernels matmul_ogs':40s}  {tk_us:8.1f} μs  ({tk_us/fastest:.2f}x)")
    print(f"    {'Native grouped_mm (compiled)':40s}  {native_us:8.1f} μs  ({native_us/fastest:.2f}x)")
    print(f"    {'Native grouped_mm v2 (view+sum)':40s}  {native_v2_us:8.1f} μs  ({native_v2_us/fastest:.2f}x)")


if __name__ == "__main__":
    main()
