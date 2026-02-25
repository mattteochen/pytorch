"""Directly compile a Triton kernel with per-peer args vs indirect ptrs
and compare PTX load instructions."""

import re
from collections import Counter
import torch
import triton
import triton.language as tl

DEVICE = "cuda"
HIDDEN = 2880
BLOCK_N = triton.next_power_of_2(HIDDEN)
WORLD_SIZE = 4


@triton.jit
def kernel_per_peer_args(
    buf0, buf1, buf2, buf3, output_ptr, N,
    WORLD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Simulates the NEW codegen: per-peer TensorArgs with divisibility hints."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    idx = row * N + cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc += tl.load(buf0 + idx, mask=mask, other=0.0).to(tl.float32)
    acc += tl.load(buf1 + idx, mask=mask, other=0.0).to(tl.float32)
    acc += tl.load(buf2 + idx, mask=mask, other=0.0).to(tl.float32)
    acc += tl.load(buf3 + idx, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + idx, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def kernel_indirect_ptrs(
    buf_ptrs, output_ptr, N,
    WORLD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Simulates the OLD codegen: single int64 pointer-of-pointers."""
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


def count_instructions(ptx: str):
    counts = Counter()
    for line in ptx.split("\n"):
        s = line.strip()
        m = re.match(r"(?:@%\w+\s+)?((?:ld|st)\.global\S+)", s)
        if m:
            counts[m.group(1)] += 1
    return counts


def analyze(name, ptx):
    counts = count_instructions(ptx)
    ld_counts = {k: v for k, v in counts.items() if k.startswith("ld.") and "global" in k}
    vec_ld = sum(v for k, v in ld_counts.items() if ".v4" in k or ".v2" in k)
    scalar_ld = sum(v for k, v in ld_counts.items() if ".v4" not in k and ".v2" not in k)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Vectorized loads: {vec_ld}")
    print(f"  Scalar loads:     {scalar_ld}")
    for k, v in sorted(ld_counts.items()):
        tag = " (VECTORIZED)" if ".v4" in k or ".v2" in k else ""
        print(f"    {k:40s} {v:4d}{tag}")
    return vec_ld, scalar_ld


def main():
    device = torch.device(DEVICE)
    buf = torch.randn(1, HIDDEN, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(buf)
    buf_ptrs = torch.tensor(
        [buf.data_ptr()] * WORLD_SIZE, dtype=torch.int64, device=device,
    )

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM: {torch.cuda.get_device_capability()}")
    print(f"HIDDEN={HIDDEN}, BLOCK_N={BLOCK_N}, WORLD_SIZE={WORLD_SIZE}")

    # NEW codegen: per-peer args
    ck = kernel_per_peer_args[(1,)](
        buf, buf, buf, buf, output, HIDDEN,
        WORLD_SIZE=WORLD_SIZE, BLOCK_N=BLOCK_N,
    )
    vec_new, scalar_new = analyze("NEW: Per-peer TensorArgs (this PR)", ck.asm["ptx"])

    # OLD codegen: indirect pointers
    ck = kernel_indirect_ptrs[(1,)](
        buf_ptrs, output, HIDDEN,
        WORLD_SIZE=WORLD_SIZE, BLOCK_N=BLOCK_N,
    )
    vec_old, scalar_old = analyze("OLD: Indirect pointer-of-pointers", ck.asm["ptx"])

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  OLD (indirect): {vec_old} vec + {scalar_old} scalar = {vec_old+scalar_old} total loads")
    print(f"  NEW (per-peer): {vec_new} vec + {scalar_new} scalar = {vec_new+scalar_new} total loads")
    reduction = (vec_old + scalar_old) / max(vec_new + scalar_new, 1)
    print(f"  Reduction: {reduction:.1f}x fewer load instructions")

    if vec_new > 0 and vec_old == 0:
        print(f"\n  CONFIRMED: Per-peer args enable vectorized loads")
    elif vec_new > 0:
        print(f"\n  Both have vectorized loads (per-peer is better)")
    else:
        print(f"\n  WARNING: Per-peer args did NOT produce vectorized loads")


if __name__ == "__main__":
    main()
