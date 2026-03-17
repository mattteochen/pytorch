#!/usr/bin/env python3
# Example usage:
# python3 benchmark/kernels/benchmark_qkv_gemm_rope.py --mode both
# python3 benchmark/kernels/benchmark_qkv_gemm_rope.py --mode compile --num-tokens 1 --dtype bf16
# TORCH_COMPILE_DEBUG=1 python3 benchmark/kernels/benchmark_qkv_gemm_rope.py --mode compile

import argparse
import math
import time
from typing import Optional, Tuple

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
        "--mode",
        choices=["eager", "compile", "both"],
        default="both",
        help="Which execution path to run.",
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
    parser.add_argument("--num-tokens", type=int, default=4096)
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
    return type(rotary_emb).forward_native.__wrapped__(
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
    fn,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Tuple[float, float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    outputs = None
    with torch.no_grad():
        for _ in range(warmup):
            outputs = fn(positions, hidden_states)
        maybe_synchronize(device)

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                outputs = fn(positions, hidden_states)
            end.record()
            torch.cuda.synchronize(device)
            avg_ms = start.elapsed_time(end) / iters
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                outputs = fn(positions, hidden_states)
            maybe_synchronize(device)
            avg_ms = (time.perf_counter() - t0) * 1000.0 / iters

    assert outputs is not None
    checksum = sum(t.float().sum() for t in outputs).item()
    return avg_ms, checksum, outputs


def max_abs_diff(
    eager_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    compiled_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    return max(
        (a.float() - b.float()).abs().max().item()
        for a, b in zip(eager_outputs, compiled_outputs)
    )


def build_compiled(model: nn.Module, args: argparse.Namespace):
    compile_kwargs = {
        "dynamic": args.dynamic,
        "fullgraph": args.fullgraph,
    }
    options = {
        "combo_kernels": True,
        "trace.enabled": True,
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "TRITON",
    }
    if args.compile_mode is not None:
        compile_kwargs["mode"] = args.compile_mode
    return torch.compile(model, options=options, **compile_kwargs)


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
    if args.num_tokens > args.max_position:
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

    positions, hidden_states = make_inputs(
        num_tokens=args.num_tokens,
        hidden_size=args.hidden_size,
        max_position=args.max_position,
        dtype=dtype,
        device=device,
    )

    print("=== Configuration ===")
    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"mode={args.mode}")
    print(f"hidden_states={tuple(hidden_states.shape)}")
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

    eager_outputs = None
    if args.mode in {"eager", "both"}:
        eager_ms, eager_checksum, eager_outputs = benchmark_fn(
            fn=model,
            positions=positions,
            hidden_states=hidden_states,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        print("\n=== Eager ===")
        print(f"avg_ms={eager_ms:.4f}")
        print(f"checksum={eager_checksum:.6f}")
        print(f"output_shapes={tuple(t.shape for t in eager_outputs)}")

    if args.mode in {"compile", "both"}:
        compiled_model = build_compiled(model, args)
        compiled_ms, compiled_checksum, compiled_outputs = benchmark_fn(
            fn=compiled_model,
            positions=positions,
            hidden_states=hidden_states,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        print("\n=== torch.compile ===")
        print(f"avg_ms={compiled_ms:.4f}")
        print(f"checksum={compiled_checksum:.6f}")
        print(f"output_shapes={tuple(t.shape for t in compiled_outputs)}")

        if eager_outputs is None:
            with torch.no_grad():
                eager_outputs = model(positions, hidden_states)
        diff = max_abs_diff(eager_outputs, compiled_outputs)
        print(f"max_abs_diff_vs_eager={diff:.6e}")


if __name__ == "__main__":
    main()
