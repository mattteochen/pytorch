"""
Correct PTX analysis for GB200 (SM 10.0) — handle predicated instructions.
On SM 10.0, Triton generates predicated ld/st like:
    @%p1 ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd1 + 0 ];
Our regex needs to match these too.
"""

import re
from collections import Counter

import torch
import triton
import triton.language as tl


DEVICE = "cuda"
HIDDEN = 2880
BLOCK_N = triton.next_power_of_2(HIDDEN)


@triton.jit
def kernel_direct_load(
    buf_ptr, output_ptr, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    val = tl.load(buf_ptr + row * N + cols, mask=mask, other=0.0)
    tl.store(output_ptr + row * N + cols, val, mask=mask)


@triton.jit
def kernel_runtime_ptr_load(
    buf_ptrs, output_ptr, N,
    WORLD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    idx = row * N + cols
    buf_ptrs_u64 = buf_ptrs.to(tl.pointer_type(tl.uint64))
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in tl.static_range(WORLD_SIZE):
        peer = tl.load(buf_ptrs_u64 + i).to(tl.pointer_type(tl.bfloat16))
        acc += tl.load(peer + idx, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + idx, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def kernel_direct_per_peer(
    buf0, buf1, buf2, buf3, output_ptr, N,
    WORLD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    idx = row * N + cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    if WORLD_SIZE >= 1:
        acc += tl.load(buf0 + idx, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 2:
        acc += tl.load(buf1 + idx, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 3:
        acc += tl.load(buf2 + idx, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 4:
        acc += tl.load(buf3 + idx, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + idx, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def kernel_fused_direct(
    buf0, buf1, buf2, buf3,
    output_ptr, weight_ptr, residual_ptr, residual_out_ptr,
    N, stride_row, eps,
    WORLD_SIZE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    row_off = row_idx * stride_row + cols
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if WORLD_SIZE >= 1:
        acc += tl.load(buf0 + row_off, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 2:
        acc += tl.load(buf1 + row_off, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 3:
        acc += tl.load(buf2 + row_off, mask=mask, other=0.0).to(tl.float32)
    if WORLD_SIZE >= 4:
        acc += tl.load(buf3 + row_off, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        res = tl.load(residual_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        acc = acc + res
        tl.store(residual_out_ptr + row_off, acc.to(tl.bfloat16), mask=mask)
    mean_sq = tl.sum(acc * acc, axis=0) / N
    rnorm = tl.math.rsqrt(mean_sq + eps)
    wt = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = acc * rnorm * wt
    tl.store(output_ptr + row_off, out.to(tl.bfloat16), mask=mask)


@triton.jit
def kernel_fused_indirect(
    buf_ptrs,
    output_ptr, weight_ptr, residual_ptr, residual_out_ptr,
    N, stride_row, eps,
    WORLD_SIZE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    row_off = row_idx * stride_row + cols
    buf_ptrs_u64 = buf_ptrs.to(tl.pointer_type(tl.uint64))
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in tl.static_range(WORLD_SIZE):
        peer = tl.load(buf_ptrs_u64 + i).to(tl.pointer_type(tl.bfloat16))
        acc += tl.load(peer + row_off, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        res = tl.load(residual_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        acc = acc + res
        tl.store(residual_out_ptr + row_off, acc.to(tl.bfloat16), mask=mask)
    mean_sq = tl.sum(acc * acc, axis=0) / N
    rnorm = tl.math.rsqrt(mean_sq + eps)
    wt = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = acc * rnorm * wt
    tl.store(output_ptr + row_off, out.to(tl.bfloat16), mask=mask)


def count_instructions(ptx: str):
    """Count all ld.global and st.global instructions including predicated ones."""
    counts = Counter()
    for line in ptx.split("\n"):
        s = line.strip()
        # Match both unpredicated and predicated: @%p1 ld.global.v4.b32 ...
        m = re.match(r'(?:@%\w+\s+)?((?:ld|st)\.global\S+)', s)
        if m:
            counts[m.group(1)] += 1
    return counts


def print_analysis(name: str, ptx: str):
    counts = count_instructions(ptx)
    ptx_lines = len(ptx.splitlines())

    ld_counts = {k: v for k, v in counts.items() if k.startswith("ld.")}
    st_counts = {k: v for k, v in counts.items() if k.startswith("st.")}

    vec_ld = sum(v for k, v in ld_counts.items() if ".v4" in k or ".v2" in k)
    scalar_ld = sum(v for k, v in ld_counts.items() if ".v4" not in k and ".v2" not in k)
    vec_st = sum(v for k, v in st_counts.items() if ".v4" in k or ".v2" in k)
    scalar_st = sum(v for k, v in st_counts.items() if ".v4" not in k and ".v2" not in k)

    verdict = "VECTORIZED" if vec_ld > 0 else "SCALAR ONLY"

    print(f"\n{'=' * 70}")
    print(f"  {name}: {verdict}")
    print(f"{'=' * 70}")
    print(f"  PTX lines: {ptx_lines}")
    print(f"  Loads:  vec={vec_ld:3d}  scalar={scalar_ld:3d}  total={vec_ld+scalar_ld}")
    print(f"  Stores: vec={vec_st:3d}  scalar={scalar_st:3d}  total={vec_st+scalar_st}")
    if ld_counts:
        print(f"  Load breakdown:")
        for k, v in sorted(ld_counts.items()):
            tag = " (vectorized)" if ".v4" in k or ".v2" in k else ""
            print(f"    {k:45s} {v:4d}{tag}")
    if st_counts:
        print(f"  Store breakdown:")
        for k, v in sorted(st_counts.items()):
            tag = " (vectorized)" if ".v4" in k or ".v2" in k else ""
            print(f"    {k:45s} {v:4d}{tag}")
    return verdict, vec_ld, scalar_ld


def main():
    device = torch.device(DEVICE)
    N = HIDDEN
    buf = torch.randn(1, N, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(buf)
    weight = torch.ones(N, device=device, dtype=torch.float32)
    residual = torch.randn_like(buf)
    residual_out = torch.empty_like(buf)
    WORLD_SIZE = 4
    buf_ptrs = torch.tensor(
        [buf.data_ptr()] * WORLD_SIZE, dtype=torch.int64, device=device,
    )

    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM: {torch.cuda.get_device_capability()}")
    print(f"HIDDEN={N}, BLOCK_N={BLOCK_N}, WORLD_SIZE={WORLD_SIZE}")
    print("=" * 70)

    grid = (1,)
    results = {}

    # Test 1: Direct buffer pointer (baseline)
    ck = kernel_direct_load[grid](buf, output, N, BLOCK_N=BLOCK_N)
    results["1_direct"] = print_analysis(
        "Direct buffer ptr (baseline)", ck.asm["ptx"])

    # Test 2: Runtime pointer indirection (current codegen)
    ck = kernel_runtime_ptr_load[grid](
        buf_ptrs, output, N, WORLD_SIZE=WORLD_SIZE, BLOCK_N=BLOCK_N)
    results["2_indirect"] = print_analysis(
        "Runtime ptr indirection (CURRENT CODEGEN)", ck.asm["ptx"])

    # Test 3: Direct per-peer args (proposed fix)
    ck = kernel_direct_per_peer[grid](
        buf, buf, buf, buf, output, N,
        WORLD_SIZE=WORLD_SIZE, BLOCK_N=BLOCK_N)
    results["3_per_peer"] = print_analysis(
        "Direct per-peer args (PROPOSED FIX)", ck.asm["ptx"])

    # Test 4: Fused AR+RMSNorm with direct args
    ck = kernel_fused_direct[grid](
        buf, buf, buf, buf, output, weight, residual, residual_out,
        N, N, 1e-5, WORLD_SIZE=WORLD_SIZE, HAS_RESIDUAL=True, BLOCK_N=BLOCK_N)
    results["4_fused_direct"] = print_analysis(
        "Fused AR+RMSNorm DIRECT args", ck.asm["ptx"])

    # Test 5: Fused AR+RMSNorm with indirect ptr
    ck = kernel_fused_indirect[grid](
        buf_ptrs, output, weight, residual, residual_out,
        N, N, 1e-5, WORLD_SIZE=WORLD_SIZE, HAS_RESIDUAL=True, BLOCK_N=BLOCK_N)
    results["5_fused_indirect"] = print_analysis(
        "Fused AR+RMSNorm INDIRECT ptr (CURRENT)", ck.asm["ptx"])

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Kernel':<50s} {'Verdict':<15s} {'Vec LD':>7s} {'Scalar LD':>10s}")
    print(f"  {'-'*50} {'-'*15} {'-'*7} {'-'*10}")
    for name, (verdict, vec, scalar) in results.items():
        print(f"  {name:<50s} {verdict:<15s} {vec:>7d} {scalar:>10d}")

    # Instruction savings
    print(f"\n  Instruction count comparison:")
    if "2_indirect" in results and "3_per_peer" in results:
        _, vec_i, sc_i = results["2_indirect"]
        _, vec_p, sc_p = results["3_per_peer"]
        total_i = vec_i + sc_i
        total_p = vec_p + sc_p
        print(f"    Simple reduce-load: indirect={total_i} instrs, per-peer={total_p} instrs")
    if "5_fused_indirect" in results and "4_fused_direct" in results:
        _, vec_i, sc_i = results["5_fused_indirect"]
        _, vec_p, sc_p = results["4_fused_direct"]
        total_i = vec_i + sc_i
        total_p = vec_p + sc_p
        print(f"    Fused AR+RMSNorm:  indirect={total_i} instrs, per-peer={total_p} instrs")


if __name__ == "__main__":
    main()
