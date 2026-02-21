# Inductor-Generated Fused AllReduce via Kraken PTX

## Status: Working E2E on 4xGB200, matching kraken perf

All 22 tests pass (14 single-process + 6 multi-GPU distributed + 2 torch.compile e2e).
The `torch.compile` path generates a single fused Triton kernel that does
P2P allreduce + residual add + RMSNorm with kraken device-side sync.

Profiled on 4xGB200 with NUM_TOKENS=32, HIDDEN=2880:
- **compiled** (inductor P2P): 1.008ms total — matches kraken (1.002ms)
- Beats baseline NCCL (4.5ms) by 4.5x
- Beats handwritten fused_op with host barriers (1.4ms) by 1.4x

## Pipeline

```
User code:  funcol.all_reduce(x) → x + residual → F.rms_norm(x, w, eps)
                │
                ▼  FX pass (post_grad.py, config._fused_all_reduce_rmsnorm)
Transformed:  symm_mem.p2p_allreduce(x) → add(_, residual) → decomposed rmsnorm
                │
                ▼  IR lowering (comm_lowering.py)
IR:           SymmMemP2PAllReduce Pointwise → add Pointwise → Reduction (rmsnorm)
                │
                ▼  Scheduler fusion (all fuse into one kernel)
                │
                ▼  Triton codegen
Generated:    prologue (copy→symm_mem + sync) → P2P reduce load loop →
              add + pow + mean + rsqrt + mul + mul_weight → epilogue (sync)
```

## Files Modified / Created

| File | Role |
|------|------|
| `torch/_inductor/ops_handler.py` | `symm_mem_p2p_reduce_load` op on OpsHandler |
| `torch/_inductor/codegen/common.py` | CSEProxy + Kernel base method |
| `torch/_inductor/codegen/triton.py` | P2P load codegen, prologue/epilogue, kraken import, constants, call_kernel, persistent reduction override |
| `torch/_inductor/codegen/triton_utils.py` | `is_unaligned_buffer` safety for virtual buffers |
| `torch/_inductor/ir.py` | `SymmMemP2PAllReduce` factory (creates fusible Pointwise) |
| `torch/_inductor/fx_passes/fused_allreduce_rmsnorm.py` | Simplified FX pass (only replaces all_reduce+wait) |
| `torch/_inductor/comm_lowering.py` | Lowering for `symm_mem.p2p_allreduce` |
| `torch/_inductor/config.py` | `_symm_mem_skip_prologue_copy` config flag |
| `torch/_inductor/dtype_propagation.py` | Dtype rule for `symm_mem_p2p_reduce_load` |
| `torch/_inductor/dependencies.py` | Read-dep tracking |
| `torch/_inductor/loop_body.py` | Index forwarding |
| `torch/_inductor/sizevars.py` | Index simplification forwarding |
| `torch/distributed/_symmetric_memory/_p2p_allreduce.py` | Custom op (Meta + fallback) |
| `torch/distributed/_symmetric_memory/__init__.py` | Import registration |
| `torch/_inductor/runtime/symm_mem_helpers.py` | Wrapper runtime: cached workspace + pointer tensors |
| `test/distributed/test_fused_allreduce_rmsnorm.py` | 22 tests (pattern, op, compile e2e, multi-GPU) |
| `dev/profile_fused_allreduce_rmsnorm.py` | 8-variant profiling script with --nsys mode |

## Generated Kernel vs Kraken Reference

| Aspect | Generated | Kraken handwritten |
|--------|-----------|-------------------|
| Copy-in | `tl.load(buffer_ptrs + RANK)` → `tl.store` | Same |
| Sync | `_symm_mem_sync(signal_pad_ptrs, ...)` | Same function |
| P2P reduce | `tl.static_range(WS)` loop, pointer-to-pointer deref | `range(world_size)` loop, same deref |
| RMSNorm | **Inductor-generated** (pow→mean→rsqrt→mul) | Hand-coded |
| Grid | Heuristic-chosen XBLOCK, prologue loops over XBLOCK rows | `num_blocks` (one per row) |
| Reduction | Persistent (forced for P2P kernels) | Persistent (hardcoded) |
| Kernel launches | 1 | 1 |

## Profiling Results (4xGB200, HIDDEN=2880, CUDA graphs)

### NUM_TOKENS=32

| Variant | Fused kernel (avg) | Total Self CUDA | Launches |
|---------|-------------------|-----------------|----------|
| baseline (NCCL) | 149us (NCCL) | 4.5ms | many |
| fused_op (handwritten, host barriers) | 10.8us | 1.4ms | 3 |
| **compiled** (inductor P2P) | 18.5us | **1.008ms** | **1** |
| compiled_mempool (inductor P2P + zero-copy) | 25.1us | 1.07ms | 1 |
| mempool (handwritten + zero-copy) | 11.0us | 1.3ms | 3 |
| **kraken** (handwritten reference) | 15.1us | **1.002ms** | **1** |
| flashinfer (TRT-LLM) | 9.2us | 837us | 1 |

### NUM_TOKENS=1

| Variant | Fused kernel (avg) | Total Self CUDA |
|---------|-------------------|-----------------|
| baseline (NCCL) | 180us | 3.7ms |
| fused_op | 10.8us | 1.4ms |
| **compiled** | 25.8us | **1.15ms** |
| compiled_mempool (skip_copy) | 25.1us | 1.14ms |
| kraken | 17.1us | 982us |
| flashinfer | 8.7us | 885us |

### 2xGB200 wall-clock benchmarks (--timer mode, CUDA graphs)

NUM_TOKENS=1:

| Variant | us/iter | vs baseline |
|---------|---------|-------------|
| baseline (NCCL) | 29.1 | 1.00x |
| fused_op (host barriers) | 14.3 | 2.04x |
| **compiled** (device sync) | 11.7 | 2.49x |
| compiled_plain (no fusion) | 16.8 | 1.73x |
| kraken (device sync) | 11.2 | 2.58x |
| flashinfer (Lamport) | 5.2 | 5.55x |

NUM_TOKENS=1024:

| Variant | us/iter | vs baseline |
|---------|---------|-------------|
| baseline (NCCL) | 86.8 | 1.00x |
| fused_op (host barriers) | 24.4 | 3.56x |
| **compiled** (device sync) | 49.8 | 1.74x |
| compiled_plain (no fusion) | 51.2 | 1.70x |
| mempool (host barriers) | 23.9 | 3.63x |
| kraken (device sync) | 75.1 | 1.16x |
| flashinfer (Lamport, forced one-shot) | 35.8 | 2.42x |

Key takeaway: at 1 token, device-side sync wins (saves barrier launch
overhead). At 1024 tokens, device-side per-block atomics scale badly —
host barriers and Lamport are both significantly faster. See "Sync
Strategy Analysis" below.

## Sync Strategy Analysis

Three synchronization approaches are in play, with very different scaling:

| Approach | Mechanism | Cost scaling | Best for |
|----------|-----------|-------------|----------|
| **Device-side atomics** (compiled, kraken) | Per-block `atom.global.sys.cas` over NVLink | O(num_blocks) system-scope atomics | Small grids (1–32 tokens) |
| **Host barriers** (fused_op, mempool) | 2 `symm_mem.barrier()` kernel launches | O(1) — fixed 2 launches | Large grids (many tokens) |
| **Lamport sentinel** (FlashInfer) | Push data to all peers, poll local memory for `-0.0` sentinel | O(data_size) local volatile loads | Moderate grids; no atomics at all |

At 1 token (tiny grid), device-side sync wins by eliminating 2 barrier
launches (~3us savings). At 1024 tokens, each block does 2 system-scope
CAS operations — that's ~2048 NVLink atomics with world_size=2,
overwhelming any launch-latency savings. Host barriers stay at exactly 2
launches regardless of grid size.

FlashInfer's Lamport protocol avoids both per-block atomics AND extra
kernel launches. It uses a write-to-all (push) model with `-0.0` as a
sentinel: each rank writes its data to all peers' local buffers, then
receivers poll their own local memory via volatile loads until the data
is no longer `-0.0`. The sync cost is per-element local loads (cheap)
rather than per-block NVLink atomics (expensive).

FlashInfer also auto-switches between one-shot (≤128 tokens) and
two-shot (>128 tokens) via `use_oneshot(token_num)`. Two-shot does
reduce-scatter + allgather with only 2 device-side barriers, halving
NVLink traffic. The profiling script was forcing `use_oneshot=True`;
changed to `use_oneshot=None` to let the kernel auto-tune.

### Why persistent reduction is correct

The forced persistent override is the right choice for P2P kernels.
Without it, the heuristic would choose looped reduction for
`r0_numel=2880 > threshold=1024`, causing P2P loads to repeat per loop
chunk: `world_size × num_chunks` NVLink loads instead of `world_size`.
The handwritten kernel (`fused_op`) also uses persistent-style reduction
(`BLOCK_N = next_power_of_2(N)`, one block handles the full row).

The performance gap at 1024 tokens (`compiled` 49.8us vs `fused_op`
24.4us) is NOT caused by the reduction strategy — both are persistent.
It's caused by the device-side sync scaling (see above). Evidence:
`compiled` (49.8us) is nearly identical to `compiled_plain` (51.2us,
no P2P fusion at all), confirming the P2P fusion overhead is minimal
and the bottleneck is the sync model.

### Post-kernel barrier purpose

The second barrier (or epilogue device-side sync) is NOT needed for the
norm computation — each rank computes norm independently after reducing.
It protects against a **read-after-write hazard across iterations**:
without it, rank 0 could start iteration N+1's copy into its symmetric
buffer while rank 1 is still reading from rank 0's buffer in iteration N.
Double buffering (alternating between two symmetric buffers) would
eliminate this barrier.

## Completed Optimizations

### [DONE] Heuristic-chosen XBLOCK (not forced to 1)

Previously XBLOCK was forced to 1 for grid consistency. Fixed by making
the prologue loop over XBLOCK rows (`for _symm_row in tl.static_range(XBLOCK)`).
The heuristic now freely chooses XBLOCK. Grid consistency is guaranteed
because all ranks see the same tensor shapes → same heuristic output.

### [DONE] Persistent reduction forced for P2P kernels

The heuristic chose looped reduction for HIDDEN=2880 (r0_numel > 2048),
causing the P2P loads to repeat on every loop iteration — 4 peers × 6
chunks = 24 NVLink loads instead of 4. Fixed by overriding the heuristic:

```python
if kernel_features.contains_op("symm_mem_p2p_reduce_load"):
    kernel_kwargs["override_persistent_reduction"] = True
```

This matches kraken's approach (BLOCK_SIZE = next_power_of_2(D)).
Result: kernel went from `triton_red_` (24us) to `triton_per_` (18us).

### [DONE] Mempool zero-copy + skip prologue copy

Added `torch._inductor.config._symm_mem_skip_prologue_copy` flag.
When the upstream matmul writes directly to symmetric memory via
`symm_mem.get_mem_pool()`, the prologue copy is redundant. Setting
this flag via `torch.compile(options={"_symm_mem_skip_prologue_copy": True})`
generates a kernel that syncs only (no copy).

### [DONE] --nsys profiling mode

The profiling script (`dev/profile_fused_allreduce_rmsnorm.py`) supports
`--nsys` flag which skips torch profiler and emits NVTX ranges instead:

```bash
# torch profiler (default):
torchrun --nproc_per_node=4 dev/profile_fused_allreduce_rmsnorm.py

# nsys with NVTX markers:
nsys profile torchrun --nproc_per_node=4 dev/profile_fused_allreduce_rmsnorm.py --nsys
```

### [DONE] FlashInfer comparison variant

Added FlashInfer `trtllm_allreduce_fusion` (no quant) as variant 8
in the profiling script for direct comparison. Changed from forced
`use_oneshot=True` to `use_oneshot=None` so FlashInfer auto-selects
one-shot (≤128 tokens) vs two-shot (>128 tokens) based on its internal
heuristic (`kOneShotMaxToken = 128` in the C++ kernel).

## Known Limitation: Large Token Counts

The current implementation uses **one-shot P2P allreduce**: every rank
reads the FULL tensor from ALL peers over NVLink.  This is optimal for
small tensors (decode, 1-32 tokens) where kernel launch latency
dominates.  For large tensors it becomes bandwidth-bound and loses to
NCCL and FlashInfer:

```
NVLink traffic per rank (hidden=2880):
                    1 token (5.6KB)     1024 tokens (5.6MB)
one-shot P2P:       4 × 5.6KB = 22KB   4 × 5.6MB  = 22.4MB
NCCL ring:          2 × 5.6KB = 11KB   2 × 5.6MB  = 11.2MB
TRT-LLM two-shot:  2 × 5.6KB = 11KB   2 × 5.6MB  = 11.2MB
```

Benchmark results (hidden=2880, 2xGB200, `dev/benchmark.py --no-quant`):

**NOTE:** Earlier benchmark.py results were invalid — the P2P fusion was
not activating due to the import ordering bug (see "Known Bug" above).
Corrected results with `import torch.distributed._symmetric_memory`
before any `torch.compile`:

| seq_len | inductor P2P | funcol NCCL compiled | SGLang custom AR | flashinfer one-shot |
|---------|-------------|---------------------|------------------|---------------------|
| 1       | 0.011ms     | 0.017ms             | 0.009ms          | 0.006ms             |

P2P fusion now beats NCCL compiled (11 vs 17 us) but trails SGLang's
custom allreduce (9 us) and FlashInfer (6 us) at seq_len=1.

Profile script results (`dev/profile_fused_allreduce_rmsnorm.py`,
2xGB200, CUDA graphs) show the full picture across token counts:

| tokens | compiled (P2P) | compiled_plain (NCCL) | fused_op (host barrier) | flashinfer |
|--------|---------------|-----------------------|------------------------|------------|
| 1      | 11.3us        | 16.1us                | 13.8us                 | 4.8us      |
| 1024   | 49.8us        | 51.2us                | 24.4us                 | 35.8us     |
| 2048   | 94.5us        | 66.0us                | 33.1us                 | 39.8us     |

At large token counts our kernel is slower than NCCL because:
1. **4x NVLink reads** vs 2x for ring/two-shot algorithms
2. **Prologue copy** (5.6MB input → symm mem) inside the kernel
3. **Device-side sync scaling** — each block does 2 system-scope CAS
   atomics over NVLink; at 1024 rows this is thousands of expensive
   atomics vs FlashInfer's barrier-free Lamport protocol or fused_op's
   fixed 2 host barrier launches

The FX pass should be gated on tensor size: only replace `all_reduce +
wait` with P2P when the data is small enough for one-shot to win.  For
large tensors, let NCCL handle the allreduce as a separate kernel.

**Crossover point:** roughly where `data_size * world_size` exceeds
NVLink bandwidth gains from fusion.  On GB200 with 4 ranks this is
around 64-128KB (32-64 tokens at hidden=2880).  FlashInfer dynamically
switches between one-shot and two-shot at a similar threshold.

## Known Bug: Import Ordering for p2p_allreduce Lowering

The `p2p_allreduce` Inductor lowering may silently fail to register if
`torch.distributed._symmetric_memory` is not imported before the first
`torch.compile` call.

**Root cause:** `register_symm_mem_lowerings()` runs once at
`lowering.py` import time (triggered by the first `torch.compile`). It
checks `torch.ops.symm_mem.p2p_allreduce` with a try/except
(`comm_lowering.py:759-763`). If the `_p2p_allreduce` module hasn't
been imported yet, the op doesn't exist, `AttributeError` is caught,
and the lowering is **permanently skipped**.

**Symptom:** The FX pass replaces `all_reduce → wait_tensor` with
`symm_mem.p2p_allreduce` correctly, but Inductor treats it as an extern
kernel call (no fusion into Triton) and the eager fallback crashes with
`cudaErrorIllegalAddress` because the input isn't in symmetric memory.

**Workaround:** `import torch.distributed._symmetric_memory` before any
`torch.compile` call (this registers the `p2p_allreduce` op).

**Proper fix:** Either lazy-register the lowering (check at lowering
time, not import time), or ensure the op module is imported as part of
`register_symm_mem_lowerings()`.

Related: `enable_symm_mem_for_group(group_name)` is marked deprecated
but is still required for `is_symm_mem_enabled_for_group()` to return
True (which the FX pass checks). `get_symm_mem_workspace()` does NOT
populate `_group_name_to_store`. This needs to be reconciled.

## Remaining Overhead

### 1. `symm_mem_setup` called per kernel invocation

Cached (dict lookup) but still runs Python per call.

**Fix:** Move to `Runner.__init__` or module-level init.

### 2. Two CUDA int64 tensors for pointer arrays

Created on first call, cached after.

**Fix:** Use `SymmetricMemory.buffer_ptrs_dev` / `signal_pad_ptrs_dev`
raw ints directly, bypassing tensor creation.

### 3. bf16 hardcoded in prologue pointer cast

```python
_symm_local_buf = tl.load(_symm_bptrs + SYMM_RANK).to(tl.pointer_type(tl.bfloat16))
```

**Fix:** Use the actual input dtype from `V.graph.get_dtype(name)`.

## Gating Plan: Large Tensor Fallback

The one-shot P2P allreduce must be gated on tensor size. For large
tensors, the current implementation is slower than NCCL because of 4x
NVLink reads (vs 2x for ring) and device-side CAS that scales linearly
with grid size. The FX pass should only replace `all_reduce + wait`
with `p2p_allreduce` when the data is small enough for one-shot to win.

### Crossover analysis (2xGB200, hidden=2880)

| tokens | data size | P2P fused | NCCL compiled | winner |
|--------|-----------|-----------|---------------|--------|
| 1      | 5.6KB     | 11 µs     | 17 µs         | P2P    |
| 32     | 180KB     | ~18 µs    | ~20 µs        | P2P    |
| 1024   | 5.6MB     | 50 µs     | 51 µs         | tie    |
| 2048   | 11.2MB    | 95 µs     | 66 µs         | NCCL   |
| 4096   | 22.4MB    | 179 µs    | 93 µs         | NCCL   |

The crossover is around **1024 tokens (~5MB)** for 2 GPUs. With more
GPUs the CAS pressure grows faster, so the crossover shifts left
(fewer tokens). A conservative threshold of **1MB** covers the
common decode case (1-128 tokens) while avoiding the bad regime.

### Implementation

**Where to gate:** `fused_allreduce_rmsnorm.py::_can_replace()`

This is the right place because:
- It already checks `is_symm_mem_enabled_for_group`
- It has access to the `all_reduce_node` and can inspect tensor size
  from `node.meta["val"]`
- Returning `False` falls back to the original `all_reduce + wait`
  path (NCCL), which is correct for large tensors

```python
def _can_replace(all_reduce_node: fx.Node, wait_node: fx.Node) -> bool:
    # ... existing symm_mem_enabled check ...

    # Gate on tensor size: only use P2P for small tensors
    threshold = torch._inductor.config._fused_all_reduce_rmsnorm_max_bytes
    if threshold > 0:
        val = wait_node.meta.get("val")
        if val is not None:
            nbytes = val.numel() * val.element_size()
            if nbytes > threshold:
                log.debug("Cannot replace: tensor too large (%d bytes > %d)",
                          nbytes, threshold)
                return False
    return True
```

**Config:** Add `_fused_all_reduce_rmsnorm_max_bytes` to
`torch/_inductor/config.py` (default 1MB = 1048576). Setting to 0
disables the gate (always use P2P).

**Why not gate in the lowering or codegen?** By the time we reach
`comm_lowering.py`, the FX graph already has `p2p_allreduce` — there's
no clean way to fall back to NCCL. The FX pass is the only place where
the decision is reversible (just don't replace the node).

### Future: dynamic gating

The static threshold works for now but doesn't account for:
- Different hidden dimensions (wider models hit bandwidth limits sooner)
- Different GPU topologies (8-GPU NVSwitch vs 2-GPU NVLink)
- Varying world sizes (more GPUs = more CAS pressure)

A better approach long-term: let the FX pass always replace, but teach
the codegen to emit either one-shot P2P (small) or two-shot
reduce-scatter + allgather (large) based on size. This keeps fusion
benefits at all sizes. See "Two-shot P2P allreduce" in Medium Term.

## Improvements: Short Term

- [ ] **Gate FX pass on tensor size:** Only replace `all_reduce + wait`
      with P2P when `numel * element_size < threshold` (default 1MB).
      Implement in `_can_replace()` using `node.meta["val"]`.
      Add `_fused_all_reduce_rmsnorm_max_bytes` config.
- [ ] Move `symm_mem_setup` to graph init (one-time, not per-call)
- [ ] Use `buffer_ptrs_dev` / `signal_pad_ptrs_dev` raw ints
- [ ] Fix bf16 hardcoding → use actual input dtype

## Improvements: Medium Term

- [ ] **Two-shot P2P allreduce:** Implement reduce-scatter + allgather
      in the codegen for large tensors, matching kraken's `two_shot_`
      and FlashInfer's two-shot mode.  This halves NVLink traffic from
      `world_size × data` to `2 × data`.
- [ ] **LayerNorm variant:** FX pass already general (replaces all_reduce+wait).
      Inductor generates LayerNorm kernels automatically.
- [ ] **Training support:** Currently inference-only. Training needs
      intermediates for autograd.
- [ ] **Auto-detect mempool inputs:** Instead of config flag, detect at
      codegen time whether the input buffer is in symmetric memory and
      skip the prologue copy automatically.
- [ ] **Non-sum reduce ops:** `avg` needs `/ world_size` after reduce.

## Improvements: Long Term

- [ ] **Generalize beyond allreduce:** reduce_scatter, all_gather.
- [ ] **Multi-node via NVSHMEM:** `nvshmem.get()` instead of `tl.load`.
- [ ] **Double buffering:** Eliminate epilogue sync.
- [ ] **CUDA graph capture:** Test under `torch.cuda.graph()`.
- [ ] **Upstream to PyTorch.**
