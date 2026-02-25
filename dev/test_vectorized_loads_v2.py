"""
Deeper PTX inspection for GB200 (SM 10.0) — check all load/store
instruction patterns, not just ld.global.*
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
    buf_ptr,
    output_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    val = tl.load(buf_ptr + row * N + cols, mask=mask, other=0.0)
    tl.store(output_ptr + row * N + cols, val, mask=mask)


@triton.jit
def kernel_runtime_ptr_load(
    buf_ptrs,
    output_ptr,
    N,
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
    buf0, buf1, buf2, buf3,
    output_ptr,
    N,
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


def dump_all_instructions(ptx: str, prefix: str):
    """Extract all load/store instructions regardless of encoding."""
    counts = Counter()
    for line in ptx.split("\n"):
        s = line.strip()
        # Match various load/store patterns
        for pat in ["ld.", "st.", "cp.", "tensormap."]:
            if s.startswith(pat):
                instr = s.split()[0].rstrip(";")
                counts[instr] += 1
                break
    return counts


def main():
    device = torch.device(DEVICE)
    N = HIDDEN
    buf = torch.randn(1, N, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(buf)
    WORLD_SIZE = 4
    buf_ptrs = torch.tensor(
        [buf.data_ptr()] * WORLD_SIZE,
        dtype=torch.int64, device=device,
    )

    grid = (1,)

    print("=" * 70)
    print("GPU:", torch.cuda.get_device_name())
    print("Compute capability:", torch.cuda.get_device_capability())
    print("=" * 70)

    # Test direct load
    ck = kernel_direct_load[grid](buf, output, N, BLOCK_N=BLOCK_N)
    ptx = ck.asm["ptx"]
    print(f"\nKernel: Direct load (baseline)")
    print(f"PTX lines: {len(ptx.splitlines())}")
    counts = dump_all_instructions(ptx, "")
    for k, v in sorted(counts.items()):
        print(f"  {k:50s} {v:4d}")

    # Check if there's SASS or other ASM
    print(f"\nAvailable ASM keys: {list(ck.asm.keys())}")

    # Dump first few PTX lines with actual load/store patterns
    print("\nPTX load/store lines (first 30):")
    n = 0
    for line in ptx.splitlines():
        s = line.strip()
        if any(s.startswith(p) for p in ["ld.", "st.", "cp."]) or "load" in s.lower():
            print(f"  {s[:120]}")
            n += 1
            if n >= 30:
                break

    # Also check for any instruction with 'v4' or 'v2' anywhere
    print("\nLines containing 'v4' or 'v2':")
    for line in ptx.splitlines():
        if "v4" in line or ".v2" in line:
            print(f"  {line.strip()[:120]}")

    print("\n" + "=" * 70)
    print("Kernel: Runtime ptr (current codegen)")
    ck2 = kernel_runtime_ptr_load[grid](
        buf_ptrs, output, N, WORLD_SIZE=WORLD_SIZE, BLOCK_N=BLOCK_N,
    )
    ptx2 = ck2.asm["ptx"]
    print(f"PTX lines: {len(ptx2.splitlines())}")
    counts2 = dump_all_instructions(ptx2, "")
    for k, v in sorted(counts2.items()):
        print(f"  {k:50s} {v:4d}")

    print("\nPTX load/store lines (first 30):")
    n = 0
    for line in ptx2.splitlines():
        s = line.strip()
        if any(s.startswith(p) for p in ["ld.", "st.", "cp."]):
            print(f"  {s[:120]}")
            n += 1
            if n >= 30:
                break

    print("\n" + "=" * 70)
    print("Kernel: Direct per-peer (proposed fix)")
    ck3 = kernel_direct_per_peer[grid](
        buf, buf, buf, buf, output, N,
        WORLD_SIZE=WORLD_SIZE, BLOCK_N=BLOCK_N,
    )
    ptx3 = ck3.asm["ptx"]
    print(f"PTX lines: {len(ptx3.splitlines())}")
    counts3 = dump_all_instructions(ptx3, "")
    for k, v in sorted(counts3.items()):
        print(f"  {k:50s} {v:4d}")

    print("\nPTX load/store lines (first 30):")
    n = 0
    for line in ptx3.splitlines():
        s = line.strip()
        if any(s.startswith(p) for p in ["ld.", "st.", "cp."]):
            print(f"  {s[:120]}")
            n += 1
            if n >= 30:
                break

    # Also dump TTGIR if available
    for name, ck_obj in [("direct", ck), ("indirect", ck2), ("per_peer", ck3)]:
        if "ttgir" in ck_obj.asm:
            ttgir = ck_obj.asm["ttgir"]
            print(f"\n{'=' * 70}")
            print(f"TTGIR for {name} (first 50 lines):")
            for i, line in enumerate(ttgir.splitlines()[:50]):
                print(f"  {line}")


if __name__ == "__main__":
    main()
