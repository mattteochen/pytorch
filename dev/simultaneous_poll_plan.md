# Fuse poll iterations in Lamport allreduce (FlashInfer-style simultaneous poll)

## Goal

Replace per-peer sequential `_poll_last_word` spin-loops with a single
while-loop per row that checks all non-self peers each iteration, overlapping
latency across peers.

## Files to modify

1. `torch/_inductor/runtime/lamport_helpers.py` -- replace `_lamport_poll_rows` with `_lamport_poll_all_peers`
2. `torch/_inductor/codegen/triton.py` -- update codegen call + import emission
3. `test/distributed/test_fuse_symm_mem_comms.py` -- update codegen assertions

## Planned implementation

Same signature as `_lamport_poll_rows`, different inner loop structure:

```python
@triton.jit
def _lamport_poll_all_peers(
    my_buf_base, x_base, r0_numel, chunk, n_words,
    RANK: tl.constexpr, WORLD_SIZE: tl.constexpr,
    XBLOCK: tl.constexpr, xnumel,
):
    for row in tl.static_range(XBLOCK):
        row_idx = x_base + row
        if row_idx < xnumel:
            row_offset = row_idx * r0_numel
            _lam_done = tl.full([], 0, dtype=tl.int32)
            while _lam_done == 0:
                _lam_cnt = tl.full([], 0, dtype=tl.int32)
                for peer in tl.static_range(WORLD_SIZE):
                    if peer != RANK:
                        slot_bf16 = my_buf_base + peer * chunk + row_offset
                        slot_u32 = slot_bf16.to(tl.pointer_type(tl.uint32))
                        last_addr = slot_u32 + (n_words - 1)
                        w = _volatile_load_u32_scalar(last_addr)
                        lo = w & 0xFFFF
                        hi = (w >> 16) & 0xFFFF
                        peer_ready = ((lo != _NEG_ZERO) & (hi != _NEG_ZERO)).to(tl.int32)
                        _lam_cnt = _lam_cnt + peer_ready
                _lam_done = (_lam_cnt == WORLD_SIZE - 1).to(tl.int32)
```

Codegen changes are purely mechanical renames:
- `_lamport_poll_rows(...)` -> `_lamport_poll_all_peers(...)` (call site ~line 3773)
- `_lamport_poll_rows,` -> `_lamport_poll_all_peers,` (import ~line 5713)

Keep `_poll_last_word` -- it is still used by `_lamport_poll_and_reduce` (standalone benchmark helper).

## Issue encountered: correctness failure in `test_torch_compile_lamport_no_residual`

27/28 tests pass. The single-loop approach causes a correctness regression in
`test_torch_compile_lamport_no_residual` (96-98% element mismatch, values off
by 1-3x, suggesting the kernel reads stale sentinel data from symmetric memory).

### What was verified

- **PTX is structurally correct.** The generated PTX contains 3 `ld.volatile.global.b32`
  instructions (one per non-self peer for WORLD_SIZE=4), a proper counter accumulation
  (`add.s32` + `selp.b32`), comparison against 3 (`setp.ne.b32 %p, %r, 3`), and a
  backward branch (`@%p bra $L__BB0_2`).

- **TTIR/LLIR are structurally correct.** The `scf.while` loop has the right
  loop-carried variable pattern. Addresses are computed correctly for all 3 non-self peers.

- **The per-peer `_poll_last_word` fallback (identical to old `_lamport_poll_rows`)
  passes all 28 tests.** Confirmed by renaming the function without changing the
  inner logic.

- **A standalone Triton kernel that pre-fills buffers and uses the simultaneous
  poll works fine.** The issue only manifests in the actual multi-GPU Lamport
  protocol where peers push data asynchronously.

- **The with-residual test (`test_torch_compile_lamport_allreduce_rmsnorm`) PASSES**
  with the single-loop poll. Only the no-residual variant fails.

### Variations attempted (all still fail on the no-residual test)

1. **`all_ready &= peer_ready` bitmask pattern** -- same failure.
2. **`n_ready += peer_ready; while n_ready < WORLD_SIZE - 1`** -- same failure.
3. **Separate `_lam_done` / `_lam_cnt` variables** (done computed after inner loop) -- same failure.
4. **`_check_peer_ready` helper function** (factoring out volatile load + check) -- same failure.

### Root cause hypothesis

The issue appears to be a Triton compiler interaction with `scf.while` containing
multiple `is_pure=False` inline asm volatile loads when used inside an `scf.if`
block in a real multi-GPU kernel. The exact mechanism is unclear but may involve:

- **LLVM vectorization of volatile loads within the while-loop body.** The LLIR
  shows the compiler packs two volatile load results into a `<2 x i32>` vector
  for SIMD comparison. While semantically valid, this changes the memory access
  pattern compared to the sequential per-peer loops.

- **Interaction with the `scf.if` boundary.** The poll while-loop is inside
  `scf.if %_lam_row_mask { ... }` while the subsequent data loads are outside.
  The old code has the same TTIR structure (3 sequential `scf.while` inside
  `scf.if`, data loads outside), but the generated PTX differs: the old code
  has 3 separate while-loop blocks with explicit fall-through ordering, while
  the new code has a single while-loop. The compiler may handle memory ordering
  differently for these two patterns.

- **The with-residual test may pass by luck** -- the additional computation
  (residual add, extra output store) between the poll and the data load may
  provide enough time for NVLink writes to become visible, masking the bug.

### Next steps to try

1. **Emit the simultaneous poll as inline codegen** (like the prologue/epilogue)
   rather than a `@triton.jit` helper, to give direct control over the generated
   Triton code and avoid function-boundary optimizations.

2. **Add `tl.debug_barrier()` after the while-loop** to force a thread sync that
   may act as an implicit memory fence.

3. **Use `ld.volatile.global.acquire` instead of `ld.volatile.global`** for the
   poll loads (requires custom inline asm) to enforce acquire semantics.

4. **File a Triton compiler bug** with a minimal repro showing that `scf.while`
   with multiple `is_pure=False` inline asm ops doesn't properly enforce memory
   ordering in the generated PTX.
