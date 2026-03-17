#!/usr/bin/env python3
"""
Validate the fused GEMM+RoPE kernel against the eager (unfused) implementation.

Usage:
    python dev/rope/validate_fused.py
    python dev/rope/validate_fused.py --num-tokens 4096
"""

import argparse
import time
from typing import Tuple

import torch
import torch.nn.functional as F

from main import (
    PackedQKVGemmRope,
    build_sglang_rope,
    forward_sglang_rope_native,
    make_inputs,
    resolve_device,
    resolve_dtype,
)
from fused_gemm_rope import fused_addmm_rope


def validate(
    num_tokens: int = 1,
    hidden_size: int = 2880,
    num_heads: int = 64,
    num_kv_heads: int = 8,
    head_dim: int = 64,
    rotary_dim: int = 64,
    dtype_str: str = "bf16",
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    device = resolve_device(None)
    dtype = resolve_dtype(dtype_str, device)

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    model = PackedQKVGemmRope(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=131072,
        rope_theta=150000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=32.0,
        rope_original_max_position=4096,
        rope_beta_fast=32.0,
        rope_beta_slow=1.0,
        rope_attn_factor=1.0,
        rope_extrapolation_factor=1.0,
        rope_truncate=False,
        dtype=dtype,
        device=device,
        bias=True,
        rope_style="neox",
    ).eval()

    positions, hidden_states = make_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        max_position=131072,
        dtype=dtype,
        device=device,
    )

    # Eager reference
    with torch.no_grad():
        q_ref, k_ref, v_ref = model(positions, hidden_states)

    # Fused kernel
    with torch.no_grad():
        q_fused, k_fused, v_fused = fused_addmm_rope(
            hidden_states=hidden_states,
            weight=model.qkv_weight.data,
            bias=model.qkv_bias.data,
            cos_sin_cache=model.rotary_emb.cos_sin_cache,
            positions=positions,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
        )

    # Compare
    q_diff = (q_ref.float() - q_fused.float()).abs().max().item()
    k_diff = (k_ref.float() - k_fused.float()).abs().max().item()
    v_diff = (v_ref.float() - v_fused.float()).abs().max().item()

    max_diff = max(q_diff, k_diff, v_diff)

    print(f"=== Validation (num_tokens={num_tokens}, dtype={dtype}) ===")
    print(f"Q max_abs_diff: {q_diff:.6e}  shape={q_fused.shape}")
    print(f"K max_abs_diff: {k_diff:.6e}  shape={k_fused.shape}")
    print(f"V max_abs_diff: {v_diff:.6e}  shape={v_fused.shape}")
    print(f"Q checksum ref={q_ref.float().sum().item():.4f}  fused={q_fused.float().sum().item():.4f}")
    print(f"K checksum ref={k_ref.float().sum().item():.4f}  fused={k_fused.float().sum().item():.4f}")
    print(f"V checksum ref={v_ref.float().sum().item():.4f}  fused={v_fused.float().sum().item():.4f}")

    # For bf16 GEMM, diffs up to ~0.1 are normal due to accumulation order differences
    threshold = 0.125 if dtype == torch.bfloat16 else 0.01
    status = "PASS" if max_diff <= threshold else "FAIL"
    print(f"\nOverall max_abs_diff: {max_diff:.6e}  threshold={threshold}  -> {status}")
    return status == "PASS"


def benchmark_fused(
    num_tokens: int = 1,
    hidden_size: int = 2880,
    num_heads: int = 64,
    num_kv_heads: int = 8,
    head_dim: int = 64,
    rotary_dim: int = 64,
    dtype_str: str = "bf16",
    warmup: int = 25,
    iters: int = 100,
) -> None:
    device = resolve_device(None)
    dtype = resolve_dtype(dtype_str, device)

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    model = PackedQKVGemmRope(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=131072,
        rope_theta=150000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=32.0,
        rope_original_max_position=4096,
        rope_beta_fast=32.0,
        rope_beta_slow=1.0,
        rope_attn_factor=1.0,
        rope_extrapolation_factor=1.0,
        rope_truncate=False,
        dtype=dtype,
        device=device,
        bias=True,
        rope_style="neox",
    ).eval()

    positions, hidden_states = make_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        max_position=131072,
        dtype=dtype,
        device=device,
    )

    weight = model.qkv_weight.data
    bias = model.qkv_bias.data
    cos_sin_cache = model.rotary_emb.cos_sin_cache

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            fused_addmm_rope(hidden_states, weight, bias, cos_sin_cache, positions,
                             num_heads, num_kv_heads, head_dim, rotary_dim)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        start.record()
        for _ in range(iters):
            fused_addmm_rope(hidden_states, weight, bias, cos_sin_cache, positions,
                             num_heads, num_kv_heads, head_dim, rotary_dim)
        end.record()
    torch.cuda.synchronize()
    fused_ms = start.elapsed_time(end) / iters

    # Benchmark eager for comparison
    with torch.no_grad():
        for _ in range(warmup):
            model(positions, hidden_states)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        start.record()
        for _ in range(iters):
            model(positions, hidden_states)
        end.record()
    torch.cuda.synchronize()
    eager_ms = start.elapsed_time(end) / iters

    print(f"\n=== Benchmark (num_tokens={num_tokens}, dtype={dtype}) ===")
    print(f"Eager:  {eager_ms:.4f} ms")
    print(f"Fused:  {fused_ms:.4f} ms")
    print(f"Speedup: {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    passed = validate(num_tokens=args.num_tokens, dtype_str=args.dtype)

    if args.benchmark and passed:
        benchmark_fused(
            num_tokens=args.num_tokens,
            dtype_str=args.dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
