#!/usr/bin/env python3
# Example usage:
# python3 main.py --num-tokens 32 --dtype bf16
# python3 main.py --num-tokens 1 32 --dtype bf16 --no-cuda-graphs
# TORCH_COMPILE_DEBUG=1 python3 main.py --num-tokens 32 128 --nvtx

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone benchmark for the gpt_oss-style packed QKV GEMM "
            "followed by RoPE on Q/K."
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on. Defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Tensor dtype. Defaults to gpt-oss-20b-bf16.",
    )
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[1])
    parser.add_argument("--hidden-size", type=int, default=2880)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--rotary-dim",
        type=int,
        default=None,
        help="Rotary dimension. Defaults to head_dim.",
    )
    parser.add_argument("--max-position", type=int, default=131072)
    parser.add_argument("--rope-theta", type=float, default=150000.0)
    parser.add_argument(
        "--rope-style",
        choices=["neox", "gptj"],
        default="neox",
        help="RoPE pairing convention.",
    )
    parser.add_argument(
        "--rope-scaling-type",
        choices=["none", "yarn"],
        default="yarn",
        help="RoPE scaling mode. Defaults to gpt-oss-20b YaRN.",
    )
    parser.add_argument(
        "--rope-scaling-factor",
        type=float,
        default=32.0,
        help="RoPE scaling factor. Defaults to gpt-oss-20b YaRN.",
    )
    parser.add_argument(
        "--rope-original-max-position",
        type=int,
        default=4096,
        help="Original max position embeddings before scaling.",
    )
    parser.add_argument(
        "--rope-beta-fast",
        type=float,
        default=32.0,
        help="YaRN beta_fast parameter.",
    )
    parser.add_argument(
        "--rope-beta-slow",
        type=float,
        default=1.0,
        help="YaRN beta_slow parameter.",
    )
    parser.add_argument(
        "--rope-attn-factor",
        type=float,
        default=1.0,
        help="YaRN attention scaling factor.",
    )
    parser.add_argument(
        "--rope-extrapolation-factor",
        type=float,
        default=1.0,
        help="YaRN extrapolation factor.",
    )
    parser.add_argument(
        "--rope-truncate",
        action="store_true",
        help="Enable YaRN truncation. gpt-oss-20b defaults to False.",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        help="Optional torch.compile mode, e.g. max-autotune.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Pass dynamic=True to torch.compile.",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile.",
    )
    parser.add_argument(
        "--disable-bias",
        action="store_true",
        help="Disable QKV bias in the packed projection.",
    )
    parser.add_argument(
        "--no-cuda-graphs",
        action="store_true",
        help="Disable CUDA graph replay benchmarking on CUDA.",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Emit NVTX ranges for 2 iterations per mode after benchmarking.",
    )
    return parser.parse_args()


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_arg: Optional[str], device: torch.device) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype_arg]


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ensure_sglang_server_args() -> None:
    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def build_sglang_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    rope_theta: float,
    rope_style: str,
    rope_scaling_type: str,
    rope_scaling_factor: float,
    rope_original_max_position: int,
    rope_beta_fast: float,
    rope_beta_slow: float,
    rope_attn_factor: float,
    rope_extrapolation_factor: float,
    rope_truncate: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    ensure_sglang_server_args()

    rope_scaling = None
    if rope_scaling_type == "yarn":
        rope_scaling = {
            "rope_type": "yarn",
            "factor": rope_scaling_factor,
            "original_max_position_embeddings": rope_original_max_position,
            "beta_fast": rope_beta_fast,
            "beta_slow": rope_beta_slow,
            "attn_factor": rope_attn_factor,
            "extrapolation_factor": rope_extrapolation_factor,
            "truncate": rope_truncate,
        }
    elif rope_scaling_type != "none":
        raise ValueError(f"Unsupported rope_scaling_type: {rope_scaling_type}")

    rotary_emb = get_rope(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=rope_theta,
        is_neox_style=(rope_style == "neox"),
        rope_scaling=rope_scaling,
        dtype=dtype,
    )
    return rotary_emb.to(device)


def forward_sglang_rope_native(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return type(rotary_emb).forward_native(
        rotary_emb,
        positions,
        query,
        key,
        None,
        None,
    )


class PackedQKVGemmRope(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        rope_theta: float,
        rope_scaling_type: str,
        rope_scaling_factor: float,
        rope_original_max_position: int,
        rope_beta_fast: float,
        rope_beta_slow: float,
        rope_attn_factor: float,
        rope_extrapolation_factor: float,
        rope_truncate: bool,
        dtype: torch.dtype,
        device: torch.device,
        bias: bool = True,
        rope_style: str = "neox",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.rope_style = rope_style

        total_qkv = self.q_size + 2 * self.kv_size
        weight = torch.randn(total_qkv, hidden_size, dtype=dtype, device=device)
        weight = weight / math.sqrt(hidden_size)
        self.qkv_weight = nn.Parameter(weight)

        if bias:
            self.qkv_bias = nn.Parameter(
                torch.randn(total_qkv, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("qkv_bias", None)

        self.rotary_emb = build_sglang_rope(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            rope_theta=rope_theta,
            rope_style=rope_style,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
            rope_original_max_position=rope_original_max_position,
            rope_beta_fast=rope_beta_fast,
            rope_beta_slow=rope_beta_slow,
            rope_attn_factor=rope_attn_factor,
            rope_extrapolation_factor=rope_extrapolation_factor,
            rope_truncate=rope_truncate,
            dtype=dtype,
            device=device,
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = F.linear(hidden_states, self.qkv_weight, self.qkv_bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = forward_sglang_rope_native(self.rotary_emb, positions, q, k)
        return q, k, v


def make_inputs(
    num_tokens: int,
    hidden_size: int,
    max_position: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device) % max_position
    return positions.contiguous(), hidden_states.contiguous()


def benchmark_fn(
    fn: Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    warmup: int,
    iters: int,
) -> Tuple[float, float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    outputs = None
    with torch.no_grad():
        for _ in range(warmup):
            outputs = fn()
        maybe_synchronize(device)

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                outputs = fn()
            end.record()
            torch.cuda.synchronize(device)
            avg_ms = start.elapsed_time(end) / iters
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                outputs = fn()
            maybe_synchronize(device)
            avg_ms = (time.perf_counter() - t0) * 1000.0 / iters

    assert outputs is not None
    checksum = sum(t.float().sum() for t in outputs).item()
    return avg_ms, checksum, outputs


def max_abs_diff(
    baseline_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    compiled_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    return max(
        (a.float() - b.float()).abs().max().item()
        for a, b in zip(baseline_outputs, compiled_outputs)
    )


def build_compiled(model: nn.Module, args: argparse.Namespace):
    compile_kwargs = {
        "dynamic": args.dynamic,
        "fullgraph": args.fullgraph,
    }
    options = {
        "trace.enabled": True,
        "max_autotune_gemm": True,
    }
    if args.compile_mode is not None:
        compile_kwargs["mode"] = args.compile_mode
    return torch.compile(model, options=options, **compile_kwargs)


def build_compiled_runner(
    model: nn.Module,
    args: argparse.Namespace,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    *,
    gemm_rope_pass: bool,
):
    compiled = build_compiled(model, args)
    with torch._inductor.config.patch(gemm_rope_pass=gemm_rope_pass):
        compiled(positions, hidden_states)

    def run(
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch._inductor.config.patch(gemm_rope_pass=gemm_rope_pass):
            return compiled(positions, hidden_states)

    return run


def make_zero_arg_runner(
    fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    def run():
        return fn(positions, hidden_states)

    return run


def make_cuda_graph_runner(
    fn: Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if device.type != "cuda":
        return fn

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(5):
            outputs = fn()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_outputs = fn()

    def replay():
        graph.replay()
        return static_outputs

    return replay


def clone_outputs(
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(x.clone() for x in outputs)


def run_nvtx_trace(
    runners: list[
        tuple[
            str,
            Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ]
    ],
    device: torch.device,
    *,
    use_cuda_graphs: bool,
) -> None:
    if device.type != "cuda":
        return

    print("\n=== NVTX Trace ===")
    print("running 2 iterations per mode with NVTX ranges")
    torch.cuda.cudart().cudaProfilerStart()
    with torch.inference_mode():
        for name, runner in runners:
            for idx in range(2):
                torch.cuda.nvtx.range_push(
                    f"{name}_iter_{idx}_cudagraphs_{int(use_cuda_graphs)}"
                )
                runner()
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize(device)
    torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    rotary_dim = args.head_dim if args.rotary_dim is None else args.rotary_dim

    if rotary_dim > args.head_dim:
        raise ValueError(
            f"rotary_dim ({rotary_dim}) must be <= head_dim ({args.head_dim})"
        )
    if (
        args.rope_scaling_type == "yarn"
        and args.max_position
        != int(args.rope_original_max_position * args.rope_scaling_factor)
    ):
        raise ValueError(
            "For YaRN, expected max_position == "
            f"rope_original_max_position * rope_scaling_factor, got "
            f"{args.max_position} vs "
            f"{args.rope_original_max_position} * {args.rope_scaling_factor}"
        )
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim ({rotary_dim}) must be even")
    if args.head_dim % 2 != 0:
        raise ValueError(f"head_dim ({args.head_dim}) must be even")
    if max(args.num_tokens) > args.max_position:
        print(
            "warning: num_tokens exceeds max_position, positions will wrap modulo "
            f"{args.max_position}"
        )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = PackedQKVGemmRope(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        rotary_dim=rotary_dim,
        max_position=args.max_position,
        rope_theta=args.rope_theta,
        rope_scaling_type=args.rope_scaling_type,
        rope_scaling_factor=args.rope_scaling_factor,
        rope_original_max_position=args.rope_original_max_position,
        rope_beta_fast=args.rope_beta_fast,
        rope_beta_slow=args.rope_beta_slow,
        rope_attn_factor=args.rope_attn_factor,
        rope_extrapolation_factor=args.rope_extrapolation_factor,
        rope_truncate=args.rope_truncate,
        dtype=dtype,
        device=device,
        bias=not args.disable_bias,
        rope_style=args.rope_style,
    ).eval()

    print("=== Configuration ===")
    print(f"device={device}")
    print(f"dtype={dtype}")
    print("modes=torch.compile,compile+gemm_rope")
    print(f"cuda_graphs={device.type == 'cuda' and not args.no_cuda_graphs}")
    print(f"qkv_weight={tuple(model.qkv_weight.shape)}")
    print(
        f"q_size={model.q_size}, kv_size={model.kv_size}, "
        f"head_dim={args.head_dim}, rotary_dim={rotary_dim}"
    )
    print(
        f"rope_scaling_type={args.rope_scaling_type}, "
        f"rope_scaling_factor={args.rope_scaling_factor}, "
        f"rope_original_max_position={args.rope_original_max_position}, "
        f"rope_truncate={args.rope_truncate}"
    )
    print(f"rope_module={type(model.rotary_emb).__name__}")
    print(f"num_tokens={args.num_tokens}")

    for param in model.parameters():
        param.requires_grad_(False)

    use_cuda_graphs = device.type == "cuda" and not args.no_cuda_graphs

    summary_rows: list[dict[str, float | int]] = []

    for num_tokens in args.num_tokens:
        positions, hidden_states = make_inputs(
            num_tokens=num_tokens,
            hidden_size=args.hidden_size,
            max_position=args.max_position,
            dtype=dtype,
            device=device,
        )

        runners: list[
            tuple[
                str,
                Callable[
                    [torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                ],
            ],
        ] = [
            ("torch.compile", model),
            (
                "compile+gemm_rope",
                build_compiled_runner(
                    model,
                    args,
                    positions,
                    hidden_states,
                    gemm_rope_pass=True,
                ),
            ),
        ]

        results: list[dict[str, object]] = []
        baseline_outputs = None
        baseline_ms = None
        nvtx_runners: list[
            tuple[str, Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
        ] = []

        for name, runner in runners:
            zero_arg_runner = make_zero_arg_runner(runner, positions, hidden_states)
            bench_runner = (
                make_cuda_graph_runner(zero_arg_runner, device)
                if use_cuda_graphs
                else zero_arg_runner
            )
            nvtx_runners.append((f"{name}_tokens_{num_tokens}", bench_runner))
            avg_ms, checksum, outputs = benchmark_fn(
                fn=bench_runner,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
            outputs = clone_outputs(outputs)

            if baseline_outputs is None:
                baseline_outputs = outputs
                baseline_ms = avg_ms

            diff = (
                0.0
                if name == "torch.compile"
                else max_abs_diff(baseline_outputs, outputs)
            )
            q_diff = (
                0.0
                if name == "torch.compile"
                else (baseline_outputs[0].float() - outputs[0].float()).abs().max().item()
            )
            k_diff = (
                0.0
                if name == "torch.compile"
                else (baseline_outputs[1].float() - outputs[1].float()).abs().max().item()
            )
            v_diff = (
                0.0
                if name == "torch.compile"
                else (baseline_outputs[2].float() - outputs[2].float()).abs().max().item()
            )
            results.append(
                {
                    "name": name,
                    "avg_ms": avg_ms,
                    "checksum": checksum,
                    "outputs": outputs,
                    "max_abs_diff_vs_baseline": diff,
                    "q_diff_vs_baseline": q_diff,
                    "k_diff_vs_baseline": k_diff,
                    "v_diff_vs_baseline": v_diff,
                    "speedup_vs_baseline": (baseline_ms / avg_ms) if baseline_ms is not None else 1.0,
                }
            )

        if baseline_ms is not None:
            fused_result = next(
                result for result in results if result["name"] == "compile+gemm_rope"
            )
            summary_rows.append(
                {
                    "num_tokens": num_tokens,
                    "baseline_ms": float(baseline_ms),
                    "fused_ms": float(fused_result["avg_ms"]),
                    "speedup": float(fused_result["speedup_vs_baseline"]),
                    "max_diff": float(fused_result["max_abs_diff_vs_baseline"]),
                    "q_diff": float(fused_result["q_diff_vs_baseline"]),
                    "k_diff": float(fused_result["k_diff_vs_baseline"]),
                    "v_diff": float(fused_result["v_diff_vs_baseline"]),
                }
            )

        if args.nvtx:
            run_nvtx_trace(
                nvtx_runners,
                device,
                use_cuda_graphs=use_cuda_graphs,
            )

    if summary_rows:
        print("\n=== Final Summary ===")
        header = (
            f"{'num_tokens':>10} {'torch.compile':>14} {'gemm_rope':>12} "
            f"{'speedup':>10} {'max_diff':>12} {'q_diff':>12} "
            f"{'k_diff':>12} {'v_diff':>12}"
        )
        print(header)
        print("-" * len(header))
        for row in summary_rows:
            print(
                f"{row['num_tokens']:>10d} "
                f"{row['baseline_ms']:>14.4f} "
                f"{row['fused_ms']:>12.4f} "
                f"{row['speedup']:>10.4f} "
                f"{row['max_diff']:>12.6e} "
                f"{row['q_diff']:>12.6e} "
                f"{row['k_diff']:>12.6e} "
                f"{row['v_diff']:>12.6e}"
            )


if __name__ == "__main__":
    main()
