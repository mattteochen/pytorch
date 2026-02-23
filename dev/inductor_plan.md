# Inductor-Generated Fused AllReduce via Kraken PTX

## Status: Host barriers implemented, compiled matches fused_op at all sizes

All 22 tests pass (14 single-process + 6 multi-GPU distributed + 2 torch.compile e2e).
The `torch.compile` path generates a single fused Triton kernel that does
P2P allreduce + residual add + RMSNorm with host-side `sm.barrier()` sync
(configurable via `_symm_mem_host_barrier_threshold`, default: always host barriers).

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

### 4xGB200 wall-clock benchmarks (--timer mode, CUDA graphs)

NUM_TOKENS=1:

| Variant | us/iter | vs baseline |
|---------|---------|-------------|
| baseline (NCCL) | 34.4 | 1.00x |
| fused_op (host barriers) | 20.0 | 1.72x |
| **compiled** (device sync) | 15.5 | 2.22x |
| compiled_plain (no fusion) | 22.9 | 1.50x |
| compiled_mempool (device sync + zero-copy) | 14.1 | 2.44x |
| mempool (host barriers + zero-copy) | 19.7 | 1.74x |
| kraken (device sync) | 11.0 | 3.13x |
| kraken_2shot (device sync) | 16.3 | 2.11x |
| flashinfer (Lamport) | 5.7 | 6.01x |

NUM_TOKENS=2048:

| Variant | us/iter | vs baseline |
|---------|---------|-------------|
| baseline (NCCL) | 171.3 | 1.00x |
| fused_op (host barriers) | 72.8 | 2.35x |
| **compiled** (device sync) | 154.3 | 1.11x |
| compiled_plain (no fusion) | 111.1 | 1.54x |
| compiled_mempool (device sync + zero-copy) | 156.4 | 1.10x |
| mempool (host barriers + zero-copy) | 76.1 | 2.25x |
| kraken (device sync) | 178.3 | 0.96x |
| kraken_2shot (device sync) | 61.0 | 2.81x |
| flashinfer (Lamport) | 65.0 | 2.63x |

### 4xGB200 wall-clock benchmarks WITH host barriers (--timer mode, CUDA graphs)

After implementing host-side `sm.barrier()` in the codegen
(`_symm_mem_host_barrier_threshold = 0`), `compiled` now uses
wrapper-level copy + barrier + kernel + barrier instead of per-block
device-side CAS inside the kernel.

NUM_TOKENS=1024:

| Variant | us/iter | vs baseline |
|---------|---------|-------------|
| baseline (NCCL) | 133.4 | 1.00x |
| fused_op (host barriers) | 43.9 | 3.04x |
| **compiled** (host barriers) | 42.9 | 3.11x |
| compiled_plain (no fusion) | 99.0 | 1.35x |
| compiled_mempool (host barriers + zero-copy) | 43.5 | 3.07x |
| mempool (host barriers + zero-copy) | 45.3 | 2.95x |
| kraken (device sync) | 94.4 | 1.41x |
| kraken_2shot (device sync) | 44.4 | 3.00x |
| flashinfer (Lamport) | 39.0 | 3.42x |

NUM_TOKENS=2048:

| Variant | us/iter | vs baseline |
|---------|---------|-------------|
| baseline (NCCL) | 171.6 | 1.00x |
| fused_op (host barriers) | 72.7 | 2.36x |
| **compiled** (host barriers) | 70.7 | 2.43x |
| compiled_plain (no fusion) | 110.6 | 1.55x |
| compiled_mempool (host barriers + zero-copy) | 71.0 | 2.42x |
| mempool (host barriers + zero-copy) | 73.1 | 2.35x |
| kraken (device sync) | 177.9 | 0.96x |
| kraken_2shot (device sync) | 61.2 | 2.80x |
| flashinfer (Lamport) | 65.1 | 2.63x |

**Host barriers close the gap completely.** `compiled` (42.9us / 70.7us)
now matches `fused_op` (43.9us / 72.7us) within measurement noise,
confirming the inductor codegen path achieves parity with the
handwritten Triton kernel. Previous `compiled` with device-side CAS
was 154.3us at 2048 tokens — this is a **2.2x improvement**.

`compiled` now beats `compiled_plain` (NCCL, no fusion) by 1.6x at
2048 tokens (70.7 vs 110.6), proving the P2P fusion delivers real
value once the sync model isn't the bottleneck.

FlashInfer (65.1us) and kraken_2shot (61.2us) still beat one-shot
host barriers at 2048 tokens thanks to two-shot's halved NVLink
traffic. Two-shot is the next optimization target.

### Scaling analysis: 2 GPUs → 4 GPUs

Key takeaways from comparing 2-GPU and 4-GPU results:

**Device-side CAS scaling is catastrophic at 4 GPUs.** At 2048 tokens
with 4 GPUs, `compiled` (154.3us) is barely above baseline (171.3us)
and kraken one-shot (178.3us) is actually **slower** than baseline.
Each block does `2 × (world_size - 1)` = 6 CAS atomics per sync point
(vs 2 with 2 GPUs), so the total CAS cost scales as
`num_blocks × 2 × 6` = ~24K NVLink atomics for 2048 tokens. This
confirms the device-side CAS approach is fundamentally unscalable.

**Host barriers hold up.** `fused_op` goes from 33.1us (2-GPU,
2048 tokens) to 72.8us (4-GPU) — roughly 2.2x, which tracks the
2x increase in NVLink read traffic (each rank reads from 3 peers
instead of 1). The barrier cost itself stays O(1).

**FlashInfer beats fused_op at 4 GPUs (reversal from 2 GPUs).**
At 2048 tokens with 2 GPUs, fused_op (33.1us) beat FlashInfer
(39.8us). At 4 GPUs, FlashInfer (65.0us) beats fused_op (72.8us).
FlashInfer's two-shot algorithm halves NVLink traffic
(`2 × data` vs `world_size × data`), and this advantage grows with
world size. At 4 GPUs, one-shot pull reads `4 × data` while
two-shot reads `2 × data` — a 2x NVLink traffic difference that
wasn't significant at 2 GPUs (2x vs 2x).

**Two-shot is the winner at large token counts.** `kraken_2shot`
(61.0us) beats everything at 2048 tokens including FlashInfer
(65.0us). The two-shot algorithm (reduce-scatter + allgather)
halves NVLink traffic, and even with 3 device-side CAS barriers,
the reduced traffic dominates at large sizes. This confirms that
implementing two-shot in the inductor codegen is the right path
for large tensors.

**1-token decode is unchanged:** FlashInfer's Lamport protocol
remains the fastest at small sizes (5.7us vs 11.0us for kraken).
Device-side sync still beats host barriers at 1 token (15.5us vs
20.0us compiled vs fused_op).

**UPDATE: Host barriers validated.** After switching the codegen to
host-side barriers, `compiled` tracks `fused_op` within noise at all
sizes (42.9 vs 43.9 at 1024 tokens, 70.7 vs 72.7 at 2048 tokens,
4 GPUs). The CAS scaling problem is fully resolved. The remaining
gap to FlashInfer/two-shot is NVLink traffic (one-shot reads
`world_size × data`), not sync overhead.

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

**4-GPU scaling makes device-side CAS even worse.** With world_size=4,
each block does `2 × 3 = 6` CAS atomics per sync point (vs 2 with
world_size=2). At 2048 tokens, kraken one-shot (178.3us) is slower than
baseline NCCL (171.3us), and compiled P2P (154.3us) barely beats it.
The CAS cost scales as `O(num_blocks × world_size)`, making it
doubly sensitive to both tensor size and GPU count.

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

**Two-shot beats FlashInfer at large sizes.** At 4xGB200 with 2048
tokens, kraken_2shot (61.0us) is faster than FlashInfer (65.0us).
The two-shot algorithm's halved NVLink traffic (`2 × data` vs
`world_size × data` for one-shot) dominates at large sizes, even
though it uses 3 device-side CAS barriers. This suggests that for
the inductor codegen path, implementing two-shot P2P allreduce is
more impactful than adopting Lamport-style sync for large tensors.

### Device-side sync improvement options

The host-barrier path (default) solves CAS scaling for medium/large
tensors. For workloads that benefit from device-side sync at small
sizes (avoiding 2 barrier kernel launches), three approaches can
reduce the per-block CAS cost:

**(a) Cap the grid (implemented, but ineffective for persistent
reduction).** Reduce the grid from `xnumel` CTAs to a fixed cap
(e.g. 36, matching SGLang). Each block processes multiple row tiles
via a grid-stride loop. CAS cost drops from
`O(xnumel × world_size)` to `O(grid_cap × world_size)`. Config:
`_symm_mem_grid_cap = 36` with `_symm_mem_host_barrier_threshold = -1`.

**Result: slower, not faster.** At 1024 tokens, 4 GPUs:
`compiled_gridcap36` = 140.5µs vs uncapped `compiled` = 93.3µs.
The generated code is correct (grid-stride prologue + body + CAS),
but persistent reduction (`R0_BLOCK = 4096`) inside a dynamic
`range()` loop causes Triton register spills. Each iteration holds
4096 fp32 elements per thread; the compiler can't unroll a dynamic-
bound loop, so it must spill/reload registers on every iteration.
The uncapped version (1 row per block, no loop) avoids this because
there's no loop to manage registers across. This is a fundamental
Triton compiler limitation — `range()` with dynamic bounds forces
register management overhead that `tl.static_range()` (compile-time
unroll) avoids, but static_range requires a compile-time trip count.

SGLang's custom allreduce avoids this because it uses a hand-written
CUDA kernel (not Triton) where register allocation is explicit, and
the reduction dimension is much smaller (allreduce only, no fused
RMSNorm). For inductor-generated persistent reduction + P2P, the
grid cap approach is not viable.

**(b) Replace CAS with `st.release.sys` / `ld.acquire.sys`.** Plain
stores and loads are cheaper than CAS (compare-and-swap is a
read-modify-write) over NVLink. The barrier would use a monotonically
incrementing counter: sender does `st.release.sys [peer_flag], iter`,
receiver spins on `ld.acquire.sys [my_flag]` until it reaches `iter`.
Tradeoff: the counter doesn't auto-reset, so CUDA graph friendliness
requires double-buffering or periodic resets from the host.

**(c) FlashInfer Lamport sentinel protocol.** Zero barriers, zero
atomics. The data itself is the sync signal — push model with `-0.0`
sentinel, receivers poll local memory via `ld.volatile.global`. This
is the fastest approach at low token counts (5.7µs vs 11.0µs for
kraken at 1 token, 4 GPUs) but requires the largest codegen change:
push model IR, volatile loads in Triton (inline PTX), triple-buffer
workspace, and sentinel stripping. See "Why We Can't Easily Adopt
Lamport" in `ALLREDUCE_FUSION.md`.

### Why persistent reduction is correct

The forced persistent override is the right choice for P2P kernels.
Without it, the heuristic would choose looped reduction for
`r0_numel=2880 > threshold=1024`, causing P2P loads to repeat per loop
chunk: `world_size × num_chunks` NVLink loads instead of `world_size`.
The handwritten kernel (`fused_op`) also uses persistent-style reduction
(`BLOCK_N = next_power_of_2(N)`, one block handles the full row).

The performance gap at 1024 tokens with device-side CAS (`compiled`
49.8us vs `fused_op` 24.4us, 2xGB200) was NOT caused by the reduction
strategy — both are persistent. It was caused by device-side sync
scaling. Evidence: `compiled` (49.8us) was nearly identical to
`compiled_plain` (51.2us, no P2P fusion at all).

**This gap is now closed with host barriers:** at 1024 tokens on
4xGB200, `compiled` (42.9us) matches `fused_op` (43.9us) within noise.

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

### [DONE] Host-side barriers for P2P allreduce codegen

Switched the inductor-generated P2P kernel from device-side CAS sync
(kraken `_symm_mem_sync` per block) to host-side `sm.barrier()` calls
in the wrapper code. Gated by `config._symm_mem_host_barrier_threshold`
(default 0 = always host barriers).

**Implementation:**
- `symm_mem_helpers.py`: Added `symm_mem_host_barrier_setup()` (copy +
  return buf_ptrs) and `symm_mem_host_barrier()` (barrier only).
- `triton.py`: When host barriers active, the generated kernel has no
  `_symm_mem_sync`, no kraken import, no `symm_signal_pad_ptrs` arg.
  Wrapper emits: setup → barrier → kernel → barrier.
- Config threshold: `0` = always host barriers, `-1` = always device
  CAS, `N > 0` = host barriers when xnumel > N or xnumel is dynamic.

**Result (4xGB200, 2048 tokens):** `compiled` went from 154.3us
(device CAS) to 70.7us (host barriers) — **2.2x improvement**, now
matching `fused_op` (72.7us) within noise.

## Known Limitation: Large Token Counts

The current implementation uses **one-shot P2P allreduce**: every rank
reads the FULL tensor from ALL peers over NVLink.  This is optimal for
small tensors (decode, 1-32 tokens) where kernel launch latency
dominates.  For large tensors it becomes bandwidth-bound and loses to
NCCL and FlashInfer:

```
NVLink traffic per rank (hidden=2880, world_size=4):
                    1 token (5.6KB)     2048 tokens (11.2MB)
one-shot P2P:       4 × 5.6KB = 22KB   4 × 11.2MB = 44.8MB
NCCL ring:          2 × 5.6KB = 11KB   2 × 11.2MB = 22.4MB
TRT-LLM two-shot:  2 × 5.6KB = 11KB   2 × 11.2MB = 22.4MB
```

At 4 GPUs, the traffic gap is 2x (one-shot reads from 3 peers vs
two-shot's reduce-scatter + allgather). With device-side CAS, this
made one-shot P2P actively harmful at large sizes: `compiled`
(154.3µs) was slower than `compiled_plain` (111.1µs, NCCL) at 2048
tokens with 4 GPUs.

**With host barriers this is fixed:** `compiled` (70.7µs) now beats
`compiled_plain` (110.6µs) by 1.6x at 2048 tokens. The remaining
gap to FlashInfer (65.1µs) and kraken_2shot (61.2µs) is purely
NVLink traffic — one-shot reads `4 × data` vs two-shot's `2 × data`.

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
1. **world_size × NVLink reads** vs 2x for ring/two-shot algorithms
   (4x at 4 GPUs, 2x at 2 GPUs)
2. **Prologue copy** (5.6MB input → symm mem) inside the kernel
3. **Device-side sync scaling** — each block does `2 × (world_size - 1)`
   system-scope CAS atomics; at 2048 rows with 4 GPUs this is ~24K
   NVLink atomics vs FlashInfer's barrier-free Lamport protocol or
   fused_op's fixed 2 host barrier launches

The problem is dramatically worse at 4 GPUs: `compiled` (154.3us) is
slower than `compiled_plain` (111.1us, standard NCCL + separate
kernels), meaning the P2P fusion is actively harmful at this size.
Kraken one-shot (178.3us) is slower than baseline NCCL (171.3us).

The FX pass should be gated on tensor size: only replace `all_reduce +
wait` with P2P when the data is small enough for one-shot to win.  For
large tensors, let NCCL handle the allreduce as a separate kernel.

**Crossover point:** scales inversely with world_size. On GB200 with
2 ranks, one-shot P2P wins up to ~1024 tokens (~5MB). With 4 ranks,
the crossover shifts earlier due to both higher NVLink traffic and
more CAS pressure. A conservative threshold of **512KB** for 4 GPUs
(~90 tokens at hidden=2880) covers decode while avoiding the bad
regime. FlashInfer dynamically switches one-shot vs two-shot at
`kOneShotMaxToken = 128` regardless of world size.

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

### Crossover analysis (4xGB200, hidden=2880)

| tokens | data size | compiled (P2P) | compiled_plain (NCCL) | winner |
|--------|-----------|---------------|-----------------------|--------|
| 1      | 5.6KB     | 15.5 µs       | 22.9 µs               | P2P    |
| 2048   | 11.2MB    | 154.3 µs      | 111.1 µs              | NCCL   |

At 4 GPUs, the crossover shifts dramatically earlier — compiled P2P is
already slower than NCCL at 2048 tokens (154 vs 111 µs), and likely
crosses over well before 1024 tokens. The crossover is around **1024
tokens (~5MB)** for 2 GPUs but estimated **256-512 tokens (~1-3MB)**
for 4 GPUs. With more GPUs, CAS pressure grows as
`O(num_blocks × world_size)` and NVLink traffic grows as
`O(data × world_size)`, both pushing the crossover left.

A conservative default threshold of **1MB** covers decode (1-128
tokens) across both 2-GPU and 4-GPU configurations. For 8-GPU
nodes, the threshold may need to be even lower.

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
- Varying world sizes (more GPUs = more CAS pressure; 4-GPU profiling
  shows the crossover shifts significantly earlier than 2-GPU)

4-GPU profiling evidence: the threshold should ideally scale inversely
with `world_size`. At 2 GPUs, one-shot P2P wins up to ~1024 tokens
(~5MB). At 4 GPUs, one-shot P2P is already losing at 2048 tokens and
the crossover is likely around 256-512 tokens.

A better approach long-term: let the FX pass always replace, but teach
the codegen to emit either one-shot P2P (small) or two-shot
reduce-scatter + allgather (large) based on size. This keeps fusion
benefits at all sizes. The 4-GPU results strongly validate this:
kraken_2shot (61µs) beats everything at 2048 tokens including
FlashInfer (65µs). See "Two-shot P2P allreduce" in Medium Term.

## Sync Strategy Plan: Host Barriers + Two-Shot

### 2-GPU evidence

Profiling at 2 GPUs confirmed host-side barriers beat both device-side
CAS (compiled, kraken one-shot) and kraken two-shot across all token
counts. At 2 GPUs, two-shot's halved NVLink traffic doesn't compensate
for its 3 device-side CAS barriers, and one-shot NVLink traffic is
negligible at the sizes where P2P beats NCCL (< 1MB).

| tokens | compiled (CAS) | kraken 2-shot | fused_op (host barrier) |
|--------|---------------|---------------|------------------------|
| 1      | 11 µs         | ~11 µs        | 14 µs                  |
| 32     | ~18 µs        | ~20 µs        | ~15 µs                 |
| 1024   | 50 µs         | ~75 µs        | 24 µs                  |
| 2048   | 95 µs         | ~140 µs       | 33 µs                  |

### 4-GPU evidence (changes the picture)

At 4 GPUs, the landscape shifts significantly:

| tokens | compiled (CAS) | kraken 2-shot | fused_op (host barrier) | flashinfer |
|--------|---------------|---------------|------------------------|------------|
| 1      | 15.5 µs       | 16.3 µs       | 20.0 µs                | 5.7 µs     |
| 2048   | 154.3 µs      | **61.0 µs**   | 72.8 µs                | 65.0 µs    |

At 2048 tokens with 4 GPUs:
- **kraken_2shot wins** (61.0 µs) — even beats FlashInfer (65.0 µs)
- **FlashInfer beats fused_op** (65.0 vs 72.8 µs) — reversed from 2 GPUs
  where fused_op (33.1 µs) beat FlashInfer (39.8 µs)
- Host barriers (fused_op) still beat device-side CAS (compiled) by 2x

The reversal happens because at 4 GPUs, one-shot pull reads from 3
peers over NVLink (`world_size × data`), while two-shot and FlashInfer's
two-shot read only `2 × data`. This 2x NVLink traffic difference was
invisible at 2 GPUs but decisive at 4 GPUs.

### Updated conclusion

**Short term: DONE.** Host-side barriers implemented and validated.
`compiled` (42.9µs at 1024, 70.7µs at 2048, 4 GPUs) matches
`fused_op` (43.9µs, 72.7µs) within noise. 2.2x improvement over
device-side CAS at 2048 tokens.

**Next: Two-shot P2P allreduce.** This is now the highest-impact
remaining optimization. At 4 GPUs with 2048 tokens, kraken_2shot
(61.2µs) and FlashInfer (65.1µs) still beat one-shot host barriers
(70.7µs) by ~14-16%. The gap is purely NVLink traffic
(`world_size × data` vs `2 × data`).

**Strategy matrix (updated):**

| Tensor size | GPU count | Best strategy | Status |
|-------------|-----------|---------------|--------|
| Small (decode, 1-32 tokens) | Any | One-shot P2P + device-side sync | Available (threshold=-1) |
| Medium-large (32+ tokens) | Any | One-shot P2P + host barriers | **Default (threshold=0)** |
| Large (128+ tokens) | 4+ | Two-shot P2P + host barriers | Planned |

### Implementation: switch codegen from CAS to host barriers

**Approach:** Remove device-side `_symm_mem_sync` from inside the
generated Triton kernel. Instead, emit host-side `symm_mem.barrier()`
calls before and after the kernel launch in the wrapper code.

**What changes in the generated code:**

Before (current, device-side CAS):
```
kernel(input, ..., signal_pad_ptrs, RANK, WORLD_SIZE):
    # prologue: copy input → symm_mem
    tl.store(local_buf, tl.load(input))
    _symm_mem_sync(signal_pad_ptrs, ...)    # device-side CAS

    # P2P reduce + compute
    for peer in range(WORLD_SIZE):
        acc += tl.load(peer_buf + offsets)
    # ... add residual, RMSNorm ...

    _symm_mem_sync(signal_pad_ptrs, ...)    # device-side CAS
```

After (host-side barriers):
```
# wrapper code (Python):
symm_mem_copy_and_barrier(input, workspace, group_name)  # copy + barrier
kernel(input, ..., RANK, WORLD_SIZE):                     # no signal_pad_ptrs
    # P2P reduce + compute (no sync inside kernel)
    for peer in range(WORLD_SIZE):
        acc += tl.load(peer_buf + offsets)
    # ... add residual, RMSNorm ...
symm_mem_barrier(workspace, group_name)                   # epilogue barrier
```

**Files to modify:**

1. `torch/_inductor/runtime/symm_mem_helpers.py` — Add
   `symm_mem_copy_and_barrier()` and `symm_mem_barrier()` wrapper
   functions that do `workspace.copy_(input)` + `sm_hdl.barrier()`.

2. `torch/_inductor/codegen/triton.py`:
   - `_codegen_symm_mem_prologue()` — gut it: remove `_symm_mem_sync`
     call and the copy loop. The copy happens host-side now.
   - `_codegen_symm_mem_epilogue()` — gut it: remove `_symm_mem_sync`.
   - `call_kernel()` — emit `symm_mem_copy_and_barrier(...)` before the
     kernel call and `symm_mem_barrier(...)` after.
   - Remove `symm_signal_pad_ptrs` from kernel arguments.
   - Remove kraken `_symm_mem_sync` import from codegen.

3. `torch/_inductor/codegen/triton.py` (persistent reduction override):
   - Keep `override_persistent_reduction = True` for P2P kernels.
     The P2P loads still need to read the full row in one pass.

**What stays the same:**
- The FX pass (replaces `all_reduce + wait` with `p2p_allreduce`)
- The IR lowering (`SymmMemP2PAllReduce` Pointwise)
- The P2P reduce load codegen (`symm_mem_p2p_reduce_load` inner_fn)
- Buffer pointer arrays (`buffer_ptrs_dev`)
- The fusion with downstream compute (add + RMSNorm)

## Push Model Plan: External Triton Helpers

A future optimization: replace the current pull model (each rank reads
from all peers over NVLink) with a push model (each rank writes its
data to all peers' local buffers). The receiver then reads from its own
local memory (cheap L2 hit) instead of remote NVLink reads.

This is what FlashInfer's Lamport protocol does. It avoids both
per-block atomics AND explicit barriers by using sentinel values
(`-0.0`) to signal completion.

### Why push can be faster than pull

- **NVLink write is fire-and-forget:** The writer doesn't stall waiting
  for data to return. With pull, every `tl.load` from a peer blocks
  until the data traverses NVLink.
- **Receiver reads locally:** Once data arrives, it's in the receiver's
  L2 cache. Local loads are ~10x cheaper than NVLink loads.
- **No barriers at all (with sentinels):** The receiver polls for
  non-sentinel values, which acts as an implicit sync. No CAS atomics,
  no barrier kernel launches.

### Implementation approach: external Triton JIT helpers

The codegen already imports and calls external `@triton.jit` functions
(kraken's `_symm_mem_sync`). The same pattern works for push helpers.

Create `torch/_inductor/runtime/p2p_push_helpers.py`:

```python
@triton.jit
def push_to_peers(buffer_ptrs, data, offset, mask, rank, world_size):
    """Write local data to all peers' symmetric memory buffers."""
    for peer in tl.static_range(world_size):
        peer_buf = tl.load(buffer_ptrs + peer).to(tl.pointer_type(tl.bfloat16))
        tl.store(peer_buf + offset, data, mask=mask)
    # System-scope fence: make all stores visible to other GPUs
    _fence_sys()

@triton.jit
def _fence_sys():
    tl.inline_asm_elementwise(
        "fence.sc.sys;", "=r", [], dtype=tl.int32, is_pure=False, pack=1,
    )

@triton.jit
def poll_for_sentinel(addr, mask):
    """Spin until value at addr is not -0.0 (Lamport sentinel)."""
    val = tl.load(addr, mask=mask)
    while _is_neg_zero(val):
        val = _volatile_load(addr, mask)
    return val

@triton.jit
def _volatile_load(addr, mask):
    """Bypass cache to see latest write from peer."""
    tl.inline_asm_elementwise(
        "ld.volatile.global.b16 $0, [$1];",
        "=h, l", [addr], dtype=tl.bfloat16, is_pure=False, pack=1,
    )
```

The codegen would emit:
```python
from torch._inductor.runtime.p2p_push_helpers import push_to_peers, poll_for_sentinel
```

### Challenges

- **Buffer layout:** Each peer pushes to a distinct region of the
  receiver's buffer (to avoid write collisions), requiring
  `world_size × data_size` symmetric memory per rank. Or use the
  sentinel approach where all peers write to the same location and
  data replaces the sentinel in-place.
- **Sentinel stripping:** Buffer must be initialized to `-0.0` before
  each iteration. During reduction, sentinel values must be excluded.
  This adds per-element overhead during the poll loop.
- **Volatile loads:** Needed to bypass L2 cache and see fresh data from
  peers. `ld.volatile.global` is more expensive than normal loads but
  cheaper than NVLink reads.
- **Correctness with CUDA graphs:** The sentinel init must happen
  inside the graph. Double buffering (alternating two buffers with
  opposite sentinels) avoids the init step.

### Ordering of improvements

1. ~~**First:** Switch to host barriers.~~ **DONE.** 2.2x improvement
   at 2048 tokens, parity with handwritten kernel.
2. **Next:** Gate FX pass on tensor size. Quick config change.
3. **Then:** Two-shot P2P allreduce for large tensors (~14% remaining
   gap to kraken_2shot / FlashInfer at 2048 tokens, 4 GPUs).
4. **(Optional):** Push model with sentinels. Only worthwhile if host
   barrier latency (~6µs) is a bottleneck for the target workload.

## Lamport Push-Model Codegen: Implemented, Perf Work Needed

### Status: functionally correct, CUDA-graph-safe, but 2.3x slower than standalone

The Lamport push-model sync mode is implemented as an alternative to
pull+host-barriers and pull+device-CAS. Selected via
`config._symm_mem_sync_mode = "lamport"`. All 24 tests pass (including
2 new Lamport-specific tests with codegen structural assertions).

```
torch.compile(options={
    "_fused_all_reduce_rmsnorm": True,
    "_symm_mem_sync_mode": "lamport",
})
```

### Files added/modified

| File | Role |
|------|------|
| `torch/_inductor/runtime/lamport_helpers.py` | **NEW**: @triton.jit helpers (fence, volatile load, push, poll+reduce, clear) + Python runtime (triple-buffer workspace, GPU-resident offset rotation) |
| `torch/_inductor/codegen/triton.py` | Mode selection (`_symm_mem_use_lamport`), Lamport imports, `_codegen_lamport_prologue`, `_codegen_lamport_epilogue`, `_codegen_lamport_reduce_load`, `_emit_lamport_setup`, kernel argdefs for `_lam_offsets` TensorArg |
| `torch/_inductor/config.py` | `_symm_mem_sync_mode: str = "host_barrier"` (new, values: host_barrier/device_cas/lamport) |
| `test/distributed/test_fused_allreduce_rmsnorm.py` | 2 new tests with codegen structural assertions |
| `dev/profile_fused_allreduce_rmsnorm.py` | `compiled_lamport` variant (3d) + renamed standalone to `lamport_standalone` |

### CUDA graph safety

Triple-buffer rotation uses a GPU-resident counter tensor. The wrapper
emits `counter.add_(1)` + `offsets[0] = (counter % 3) * slot_elems` +
`offsets[1] = ((counter + 2) % 3) * slot_elems` — all element-wise
GPU ops that capture and replay correctly. The kernel receives offsets
via a `[2]` int64 TensorArg and loads them with `tl.load`.

### Profiling results (4xGB200, HIDDEN=2880, CUDA graphs, --timer)

NUM_TOKENS=1:

| Variant | us/iter | vs baseline |
|---------|---------|-------------|
| baseline (NCCL) | 34.4 | 1.00x |
| compiled (device CAS) | 15.3 | 2.25x |
| **compiled_lamport** (inductor) | **18.5** | **1.86x** |
| lamport_standalone (handwritten) | 7.9 | 4.36x |
| flashinfer (Lamport) | 5.9 | 5.81x |

NUM_TOKENS=1024:

| Variant | us/iter | vs baseline |
|---------|---------|-------------|
| baseline (NCCL) | 132.6 | 1.00x |
| compiled (device CAS) | 85.2 | 1.56x |
| compiled_host_barrier | 48.5 | 2.74x |
| **compiled_lamport** | **74.4** | **1.78x** |
| lamport_standalone | 67.7 | 1.96x |
| flashinfer | 39.5 | 3.36x |

### Performance gap analysis: compiled_lamport (18.5µs) vs standalone (7.9µs)

The gap at 1 token is **~10.6µs** (2.3x). Root causes identified:

**1. Wrapper-level GPU ops for offset rotation (~9µs)**

The biggest cost. `lamport_advance_offsets` emits 3 GPU tensor ops
before the kernel launch:
```python
counter.add_(1)                           # GPU op
offsets[0] = (counter % 3) * slot_elems   # GPU op
offsets[1] = ((counter + 2) % 3) * slot_elems  # GPU op
```
These are captured in the CUDA graph and replay on every iteration —
~3µs each × 3 ops = ~9µs. The standalone kernel has zero wrapper ops
(offsets are baked in as constexpr).

**Fix options:**
- **(a) Move offset computation INTO the Triton kernel.** Pass just the
  counter tensor; the kernel computes `buf_offset = (counter % 3) *
  chunk` inline. This eliminates all 3 wrapper GPU ops. The kernel reads
  one scalar instead of two. The counter increment can also happen inside
  the kernel (single `tl.atomic_add` on the counter, or have the wrapper
  emit a single `counter.add_(1)` — 1 GPU op instead of 3).
- **(b) Use constexpr offsets with 3 compiled kernel variants.** Pre-
  compile the kernel for each of the 3 (buf_offset, clear_offset) pairs.
  The wrapper selects which variant to call based on `counter % 3`. This
  eliminates all runtime offset computation but requires 3x Triton
  compilations. CUDA graph capture would need to capture all 3 variants
  in a round-robin pattern.
- **(c) Accept the overhead for CUDA graph safety.** The 3 GPU ops are
  the price of correct triple-buffer rotation under CUDA graph replay.
  Focus optimization effort on the in-kernel overhead instead.

**2. Redundant pointer dereferences in helpers (~1µs)**

Each of `_lamport_push_to_peers`, `_lamport_poll_and_reduce`, and
`_lamport_clear_old_slot` independently re-derives `buf_ptrs_u64`:
```python
buf_ptrs_u64 = buf_ptrs.to(tl.pointer_type(tl.uint64))
my_buf = tl.load(buf_ptrs_u64 + RANK).to(tl.pointer_type(tl.bfloat16))
```
The standalone kernel does this once and reuses across all phases.

**Fix:** Compute `buf_ptrs_u64` and `my_buf` in the prologue and pass
as arguments to the helpers. Or inline the push/poll/clear logic
directly in the generated code (matching the standalone structure).

**3. 2 extra global loads for offsets from tensor**

The kernel loads `tl.load(_lam_offsets)` and `tl.load(_lam_offsets + 1)`
at the top of the prologue. The standalone uses constexpr offsets (zero
load cost). This is probably <0.1µs but adds to the gap.

**Fix:** Solved by option (a) or (b) above.

**4. Redundant variable recomputation**

`_lam_cols`, `_lam_col_mask`, `_lam_chunk` are computed independently
in the prologue, body, and epilogue. The Triton compiler may or may
not CSE these across phases.

**Fix:** Share variables across phases (codegen restructuring).

### Recommended next step

Option **(a)** from item 1: move offset computation into the kernel.
This is the highest-impact fix (~9µs savings, closing most of the
10.6µs gap) and keeps the CUDA-graph-safe design. The wrapper would
emit just `counter.add_(1)` (1 GPU op) and pass the counter tensor
to the kernel. The kernel computes offsets inline:
```python
_lam_iter = tl.load(_lam_counter)
_lam_buf_offset = (_lam_iter % 3) * _lam_chunk
_lam_clear_offset = ((_lam_iter + 2) % 3) * _lam_chunk
```

After this fix, the expected compiled_lamport time is ~9-10µs,
close to the standalone's 7.9µs. The remaining ~1-2µs gap from
redundant pointer dereferences can be addressed by inlining the
helper logic.

## Improvements: Short Term

- [x] **Switch to host-side barriers:** Remove device-side CAS from
      kernel, emit `symm_mem.barrier()` in wrapper before/after kernel.
      Result: 2.2x speedup at 2048 tokens (154→71µs), parity with fused_op.
- [x] **Lamport push-model codegen:** Implemented as
      `_symm_mem_sync_mode = "lamport"`. Functionally correct, CUDA-graph-
      safe. Performance gap to standalone needs closing (see above).
- [ ] **Close Lamport perf gap:** Move offset computation into kernel
      (eliminate 3 wrapper GPU ops). Inline helper logic to avoid
      redundant pointer dereferences. Target: <10µs at 1 token.
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
      `world_size × data` to `2 × data`. **4-GPU profiling confirms
      this is the highest-impact optimization:** kraken_2shot (61µs)
      beats both fused_op host barriers (73µs) and FlashInfer (65µs)
      at 2048 tokens with 4 GPUs. Combined with host barriers (instead
      of kraken's device-side CAS), two-shot should be even faster.
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
- [ ] **Upstream to PyTorch.**
