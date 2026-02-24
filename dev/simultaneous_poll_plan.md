# Fuse poll iterations in Lamport allreduce (FlashInfer-style simultaneous poll)

## Goal

Replace per-peer sequential `_poll_last_word` spin-loops with a single
while-loop per row that checks all non-self peers each iteration, overlapping
latency across peers.

## Status: Root cause found, fix implemented

The correctness failure was caused by a **missing acquire fence** on the
reader side. The fix adds `fence.acquire.sys` after the poll loop in both
`_lamport_poll_rows` and the new `_lamport_poll_all_peers`. The codegen
now emits `_lamport_poll_all_peers` by default.

## Root cause: missing release-acquire pairing

### The Lamport protocol's memory ordering contract

**Writer side** (prologue):
```
tl.store(data[0..N-1])     # push data to peer's local buffer
fence.sc.sys                # system-scope release: all prior stores globally visible
```

**Reader side** (poll):
```
while ld.volatile(sentinel) == NEG_ZERO: spin   # volatile = cache bypass, NOT acquire
tl.load(data[0..N-1])                           # regular non-volatile load
```

The writer's `fence.sc.sys` is a release fence. For correctness, the reader
needs a matching **acquire fence** to form a synchronization pair. Without it,
the GPU may serve subsequent `tl.load` from stale L2 cache — the sentinel
word propagated via NVLink but the data words may not yet be visible to
non-volatile loads.

### Why `ld.volatile.global.b32` is NOT acquire

Per the PTX ISA (7.4+), `ld.volatile` maps to `ld.relaxed.sys` semantics:
it bypasses L1 cache and sees the latest value from memory, but does NOT
prevent reordering of subsequent non-volatile loads. It is NOT an acquire
operation.

### Why the sequential poll masked the bug

With `_lamport_poll_rows`, each peer's `_poll_last_word` is a separate
`@triton.jit` function call containing its own `scf.while` loop. After
inlining, LLVM sees three separate while-loop structures. The function-call
boundary acts as an **accidental optimization barrier** — LLVM doesn't hoist
the regular data loads past the while-loop exits because they're in distinct
CFG regions.

With `_lamport_poll_all_peers`, all peers' volatile loads are in a **single**
`scf.while` body. The subsequent data loads are in the same LLIR basic-block
sequence. LLVM can schedule non-volatile data loads earlier since their
addresses are loop-invariant, and without an acquire fence, the hardware has
no ordering constraint.

### Why no-residual fails but with-residual passes

With residual, additional computation between the poll and data consumption
(residual add, extra output store) adds enough clock cycles for NVLink writes
to propagate. This is a timing artifact, not a correctness guarantee.

### Why FlashInfer doesn't need an acquire fence

FlashInfer's simultaneous poll (trtllm_allreduce_fusion.cuh:1235-1244) loads
the **data itself** via `load_global_volatile` and checks those values for
the sentinel. When the loop exits, the data is already in registers — no
separate non-volatile load is needed. Our codegen polls only the last word
for sentinel status, then does separate `tl.load` calls.

## Fix

Added `_fence_acquire_sys()` (inline asm `fence.acquire.sys`) to
`lamport_helpers.py`. Both `_lamport_poll_rows` and `_lamport_poll_all_peers`
call it after their poll loops, before returning to the caller which then
does regular `tl.load` for data accumulation.

The codegen now emits `_lamport_poll_all_peers` (simultaneous poll) instead
of `_lamport_poll_rows` (sequential per-peer poll).

## Files modified

1. `torch/_inductor/runtime/lamport_helpers.py`
   - Added `_fence_acquire_sys()` primitive
   - Added `_lamport_poll_all_peers()` with simultaneous poll + acquire fence
   - Added acquire fence to `_lamport_poll_rows()` (fixes latent bug there too)
2. `torch/_inductor/codegen/triton.py`
   - Call site: `_lamport_poll_rows` → `_lamport_poll_all_peers`
   - Import: added `_lamport_poll_all_peers`
3. `test/distributed/test_fuse_symm_mem_comms.py`
   - Codegen assertions: `_lamport_poll_rows` → `_lamport_poll_all_peers`
