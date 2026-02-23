# Design Doc: Fused AllReduce + RMSNorm for Inductor

**Authors:** (authored with Claude)
**Status:** Implementation complete, validated on 4-GPU node via profiling script
**Oncall:** distributed

## Motivation

In transformer-based models using Tensor Parallelism, every layer executes:

```
all_reduce(x) → wait → x + residual → rms_norm(x, weight, eps)
```

Today Inductor lowers this as 3+ separate kernel launches:
1. NCCL `all_reduce` (opaque external call)
2. Pointwise add (residual connection)
3. RMSNorm (pow → mean → add_eps → rsqrt → mul → mul_weight)

The intermediate tensors (`reduced`, `reduced + residual`) are materialized in
global memory between launches. For a hidden dimension of 8192 and sequence
length of 4096, this is ~64 MB of unnecessary global memory traffic per layer.

**This feature fuses steps 1–3 into a single kernel** using symmetric memory,
where each rank's data is accessible to all other ranks via direct P2P loads
over NVLink.

## Design Overview

The feature has three components:

### 1. FX Pattern Matching Pass

**File:** `torch/_inductor/fx_passes/fused_allreduce_rmsnorm.py`

Inductor decomposes `F.rms_norm` into primitive ATen ops during tracing.
The pass operates on this decomposed post-grad FX graph and must match the
full chain:

```
all_reduce → wait_tensor → [add(_, residual)] → pow(x,2) → mean → add(_, eps) → rsqrt → mul(x, rsqrt) → mul(_, weight) → [convert_element_type]
                                                  ↑                                          ↑
                                              wait_tensor feeds both branches ───────────────┘
```

The key challenge is that `wait_tensor` has **two consumers** in the graph:
it feeds both `pow(x, 2)` (variance computation) and `mul(x, rsqrt)` (the
normalization multiply). The pattern matcher must discover all intermediate
nodes across both branches to verify the pattern is self-contained.

**Algorithm:**
1. Scan for `mul(_, weight)` nodes where one arg is `mul(x, rsqrt)` — the
   final RMSNorm output
2. From that node, walk backward through all args recursively until finding
   `wait_tensor`, collecting every intermediate node into a `visited` set
3. Check if an `add` node in the visited set has `wait_tensor` as one operand
   (residual pattern)
4. Verify all `wait_tensor` users are within the pattern (no external consumers)
5. Replace with `symm_mem.fused_all_reduce_rmsnorm` + `getitem(0)` / `getitem(1)`

The pass also absorbs a trailing `convert_element_type` (e.g. f32→bf16) from
the RMSNorm decomposition to avoid leaving a dangling no-op dtype cast kernel.

**Inference only:** The pass is gated on `is_inference=True` (passed from
`post_grad_passes`). In training mode, AOTAutograd saves intermediate tensors
(e.g. `rsqrt`) for backward in the return tuple. The fused op does not produce
these intermediates, so fusion would leave dangling nodes that violate
topological ordering. Inference mode has no saved tensors, so all intermediate
nodes are cleanly erasable.

**Gating:** `torch._inductor.config._fused_all_reduce_rmsnorm` (default `False`).
Wired into `post_grad.py` alongside existing passes like `micro_pipeline_tp`
and `fuse_ddp_communication`.

### 2. Custom Op with Multi-Backend Dispatch

**File:** `torch/distributed/_symmetric_memory/_fused_all_reduce_rmsnorm.py`

```python
symm_mem.fused_all_reduce_rmsnorm(
    input: Tensor, weight: Tensor,
    reduce_op: str, group_name: str,
    *, residual: Tensor? = None, eps: float = 1e-6
) -> (Tensor, Tensor?)
```

Returns `(normed, pre_norm)`:
- `normed`: the RMS-normalized output
- `pre_norm`: `reduced + residual` when residual is provided, else `None`

The second output is essential for transformer architectures where the
pre-norm value becomes the residual input to the next layer (Pre-LN pattern).

**Dispatch strategy:**

| Backend | Implementation |
|---------|---------------|
| Meta | Shape inference only |
| CUDA | Try Triton P2P kernel → fallback to decomposed ops |
| CompositeExplicitAutograd | `all_reduce` + `wait_tensor` + `add` + `F.rms_norm` (for tracing, eager, CPU) |

The CUDA path gracefully degrades: if symmetric memory isn't enabled for the
group, the kernel import fails, or the runtime raises, it falls back to the
decomposed implementation. This means the feature is safe to enable even on
systems without P2P access — the FX pass still eliminates graph-level
redundancy, and the op falls back to the same ops that would have run anyway.

### 3. Triton Kernel with Symmetric Memory P2P

**File:** `torch/distributed/_symmetric_memory/_fused_allreduce_rmsnorm_triton.py`

The kernel uses P2P-mapped symmetric memory buffers. Each rank's data is
accessible to all other ranks via direct GPU memory loads (NVLink).

**Kernel logic** (one program per row, M programs total):
```
for each row (parallelized across programs):
    acc = load(rank_0_buf[row]) + load(rank_1_buf[row]) + ... + load(rank_{W-1}_buf[row])
    if has_residual:
        pre_norm = acc + residual[row]
        store pre_norm → residual_out[row]
    else:
        pre_norm = acc
    variance = sum(pre_norm²) / N
    normed = pre_norm * rsqrt(variance + eps) * weight
    store normed → output[row]
```

The kernel takes up to 8 buffer pointers (one per rank, `_MAX_WORLD_SIZE=8`)
with `WORLD_SIZE` as a `tl.constexpr`, so unused rank branches are eliminated
at compile time. All accumulation happens in fp32 registers regardless of
input dtype.

## Synchronization: Three Approaches Compared

This is the most performance-critical design axis. Three distinct
synchronization strategies have been implemented and benchmarked, each
with fundamentally different trade-offs. All three use symmetric memory
(P2P-mapped NVLink buffers) but differ in how they coordinate ranks.

### Data flow: Pull vs Push

The three approaches split into two data-flow models:

**Pull model** (our kernel, kraken): Each rank copies its data into its
own symmetric buffer, waits for all peers to do the same (barrier), then
reads from all peers' buffers over NVLink.

```
rank 0: store(my_buf)  ──barrier──  load(peer_1_buf) ← NVLink read
rank 1: store(my_buf)  ──barrier──  load(peer_0_buf) ← NVLink read
```

**Push model** (FlashInfer): Each rank writes its data directly into
every peer's local buffer over NVLink. The receiver polls its own local
memory for data arrival.

```
rank 0: store(peer_0_local + rank0_slot)  store(peer_1_local + rank0_slot)  ← NVLink writes
rank 1: store(peer_0_local + rank1_slot)  store(peer_1_local + rank1_slot)  ← NVLink writes
        ... each rank polls its own local memory ...
```

The pull model requires an explicit barrier between "all writes done" and
"start reading." The push model eliminates this: data arrival IS readiness
(see Lamport sentinel below).

### Strategy 1: Host-side barriers (fused_op, mempool)

**Used by:** handwritten Triton kernel
(`torch/distributed/_symmetric_memory/_fused_allreduce_rmsnorm_triton.py`)

Two `symm_mem.barrier()` kernel launches bracket the compute kernel.
Each barrier is a **separate 1-block CUDA kernel** (`barrier_kernel` in
`CUDASymmetricMemory.cu`):

```cpp
// Launched as: barrier_kernel<<<1, max(warp_size, world_size), 0, stream>>>
static __global__ void barrier_kernel(uint32_t** signal_pads, ...) {
  if (threadIdx.x < world_size) {
    auto target_rank = threadIdx.x;
    // CAS loop: spin until slot is 0, atomically set to 1 (release)
    try_put_signal<memory_order_release>(
        signal_pads[target_rank] + world_size * channel + rank, ...);
    // CAS loop: spin until slot is 1, atomically reset to 0 (acquire)
    try_wait_signal<memory_order_acquire>(
        signal_pads[rank] + world_size * channel + target_rank, ...);
  }
}
```

Each CAS (`cas<Sem>(addr, expected, desired)`) is a system-scope
`cuda::atomic_ref` compare-and-swap. The auto-reset (put writes `0→1`,
wait resets `1→0`) makes it CUDA-graph friendly.

**GPU timeline:** DtoD copy + barrier + kernel + barrier = 4 items.

**Cost:** Fixed at `2 × (world_size - 1)` system-scope CAS ops per
barrier, regardless of grid size. With world_size=2, that's 2 CAS per
barrier × 2 barriers = 4 CAS total per iteration.

### Strategy 2: Device-side per-block atomics (compiled, kraken)

**Used by:** inductor-generated kernel (via kraken's `symm_mem_sync`)

Same CAS mechanism as the host barrier, but executed **inside the
compute kernel by every block** using inline PTX:

```python
# kraken/_ptx_utils/symm_mem_barrier.py
@triton.jit
def symm_mem_sync(signal_pad_ptrs, block_id, rank, world_size, ...):
    # Each block gets its own signal pad slot: block_id * world_size + rank
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    if hasPreviousMemAccess:
        tl.debug_barrier()           # __syncthreads — flush intra-block stores

    if flat_tid < world_size:        # only thread 0 (or 0..WS-1) does the sync
        _send_signal(send_addrs, "release")   # atom.global.release.sys.cas 0→1
        _wait_signal(wait_addrs, "acquire")   # atom.global.sys.acquire.cas 1→0

    if hasSubsequentMemAccess:
        tl.debug_barrier()           # __syncthreads — make acquired data visible
```

**GPU timeline:** Single kernel (prologue copy + sync + P2P loads +
reduce + norm + sync).

**Cost:** `num_blocks × 2 sync_points × 2 × (world_size - 1)` system-scope
CAS ops. With 1024 tokens, world_size=2: `1024 × 2 × 2 = 4096` CAS —
vs 4 for the host barrier approach. Each CAS is a system-scope atomic
over NVLink, which is expensive.

### Strategy 3: Lamport sentinel protocol (FlashInfer)

**Used by:** FlashInfer `trtllm_allreduce_fusion`
(`third_party/flashinfer/include/flashinfer/comm/trtllm_allreduce_fusion.cuh`)

**Zero barriers, zero atomics.** Uses the data itself as the sync signal.

**Key insight: the data IS the signal.** There is no separation between
"write data" and "signal that data is ready." In the CAS approaches,
these are two distinct operations (store data, then atomic on signal pad)
that require `release`/`acquire` ordering. In Lamport, they are the same
store — when the receiver sees non-sentinel data, it knows the write
is complete because there is nothing to reorder.

**Mechanism:**

1. Comm buffers are pre-filled with `-0.0` (negative zero, bit pattern
   `0x8000` for bf16) as sentinel.

2. Writers strip `-0.0` from real data (`remove_neg_zero`) then push
   to ALL peers' local buffers via NVLink:

```cpp
for (int idx = access_id; idx < tot_access; idx += access_stride) {
    vec_t<T, VEC_SIZE> val;
    val.load(reinterpret_cast<T*>(params.allreduce_in) + idx * VEC_SIZE);
    remove_neg_zero<T, VEC_SIZE>(val);          // guarantee no -0.0
    for (int r = 0; r < NRanks; ++r) {
        val.store(reinterpret_cast<T*>(comm.data_bufs[r]) +
                  (params.rank * tot_access + idx) * VEC_SIZE);  // NVLink write
    }
}
```

3. Receivers poll their **own local memory** via volatile loads until
   data is no longer `-0.0`:

```cpp
while (!done) {
    done = true;
    for (int r = 0; r < NRanks; ++r) {
        vals[r].load_global_volatile(               // local DRAM read, NOT NVLink
            reinterpret_cast<T*>(comm.data_bufs[params.rank]) + ...);
        done &= !has_neg_zero<T, VEC_SIZE>(vals[r]);  // sentinel check
    }
}
```

The volatile load (`ld.global.volatile`) bypasses L1 cache and reads
from L2/DRAM, ensuring the poll sees the remote write. Since the writer
stores to the receiver's local memory (via NVLink P2P mapping), the
receiver's volatile load is a cheap local DRAM read, not an NVLink
traversal.

**Triple buffering eliminates the post-iteration barrier.** The comm
flag rotates through `0 → 1 → 2 → 0`. Each iteration writes to buffer
`flag % 3` and clears buffer `(flag + 2) % 3` (from 2 iterations ago):

```
Iter 0: write buf[0], clear buf[1]  (wrap)
Iter 1: write buf[1], clear buf[2]
Iter 2: write buf[2], clear buf[0]  ← buf[0] last used in iter 0, safe
Iter 3: write buf[0], clear buf[1]  ← buf[0] last used in iter 2, safe
```

By the time a buffer is reused, no rank can still be reading from it.
The read-after-write hazard is eliminated by construction.

**Cost:** 0 kernel launches for sync, 0 system-scope atomics. Sync cost
is O(data_size) local volatile loads (cheap) + O(data_size) sentinel
clearing + NRanks NVLink writes per element. Uses 3x comm buffer memory.

### Summary: sync cost at scale

| | Pre-sync mechanism | Post-sync mechanism | Kernel launches | NVLink atomics per iter (WS=2, 1024 tokens) |
|---|---|---|---|---|
| **Host barriers** | CAS in dedicated 1-block kernel | Same | 3 | 4 (fixed) |
| **Device-side** | CAS per block (inline PTX) | Same | 1 | ~8192 (scales with grid) |
| **Lamport** | Sentinel poll (local volatile load) | **none** (triple buffering) | 1 | 0 |

**Benchmark results (2xGB200, HIDDEN=2880, CUDA graphs):**

| Approach | 1 token | 1024 tokens | Scaling |
|----------|---------|-------------|---------|
| Device-side atomics (compiled) | **11.7us** | 49.8us | Poor — O(blocks) NVLink atomics |
| Host barriers (fused_op) | 14.3us | **24.4us** | Good — O(1) barrier cost |
| Lamport (FlashInfer, one-shot) | **5.2us** | 35.8us | Moderate — O(data) local polls |

At 1 token, device-side sync wins by eliminating 2 barrier launches
(~3us savings). At 1024 tokens, per-block atomics overwhelm any
launch-latency savings.

### FlashInfer one-shot vs two-shot

FlashInfer auto-switches at `kOneShotMaxToken = 128`:
- **One-shot** (≤128 tokens): Lamport sentinel protocol, all-to-all push
- **Two-shot** (>128 tokens): Reduce-scatter + allgather with only 2
  device-side barriers (using `st.global.release.sys`/`ld.global.acquire.sys`),
  halving NVLink traffic from `NRanks × data` to `2 × data`

The profiling script passes `use_oneshot=None` to let FlashInfer
auto-select.

### Post-iteration barrier purpose

The second sync (host barrier or device-side epilogue) is NOT needed for
the norm computation — each rank computes norm independently after reducing.
It protects against a **read-after-write hazard across iterations**: without
it, rank 0 could start iteration N+1's copy into its symmetric buffer while
rank 1 is still reading from rank 0's buffer in iteration N.

Three ways to eliminate this:
- **Double buffering:** Alternate between two symmetric buffers. Cost: 2x
  comm buffer memory.
- **Triple buffering (FlashInfer):** Rotate through 3 buffers, clear the
  one from 2 iterations ago. Cost: 3x memory, but also eliminates the
  pre-iteration barrier via the sentinel protocol.
- **Separate output buffer:** If the reduced result is written to a
  non-symmetric output (not back to the comm buffer), reads and writes
  don't conflict. Doesn't help when the comm buffer IS the next
  iteration's input.

## Why We Can't Easily Adopt Lamport

FlashInfer's Lamport protocol is the fastest approach at small token
counts and avoids all barriers. Adopting it in the inductor codegen
would require addressing five architectural gaps:

### 1. Pull → Push model change

The entire codegen pipeline is built around pull: `symm_mem_p2p_reduce_load`
emits `tl.load` from peer buffers. Lamport requires push: each rank
stores to all peers' local buffers, then polls local memory. This is a
fundamentally different IR op and data flow through the scheduler.

```
Current (pull):   prologue: copy → my_buf, SYNC → tl.load(peer_buf) → reduce
Lamport (push):   tl.store(peer_0_local), tl.store(peer_1_local) → poll_local → reduce
```

### 2. No volatile loads in Triton

FlashInfer uses `load_global_volatile` (CUDA C++) to bypass L1 cache
during the sentinel poll. Triton has no `tl.load` volatile mode — the
compiler may hoist or cache the load, breaking the spin loop. Inline
PTX would work but is awkward for per-element polls in generated code.

### 3. Triple-buffer infrastructure

FlashInfer manages its own IPC workspace with 3x comm buffers and a
rotating flag (`flag_value % 3`). PyTorch's symmetric memory API
(`get_symm_mem_workspace`) provides a single buffer with explicit
barriers. Supporting triple buffering requires changes to the
`SymmetricMemory` C++ class and the `symm_mem_helpers.py` runtime.

### 4. Sentinel value constraint

The `-0.0` trick only works for floating-point types where:
- `-0.0` is bitwise distinct from `+0.0` and all other values
- Real data can reliably avoid `-0.0` (`remove_neg_zero` strips it)
- The sentinel check is cheap (bitwise comparison)

This doesn't generalize to integer types. FlashInfer handles FP4/FP8
quant by quantizing AFTER the reduce, so the comm buffer is always in
the original float type.

### 5. What it would take

| Change | Files affected |
|--------|---------------|
| Push-model IR op (store-to-all-peers + poll-local) | `ir.py`, `ops_handler.py`, `codegen/triton.py` |
| Volatile load in Triton (inline PTX) | `codegen/triton.py` P2P load codegen |
| Triple-buffered workspace | `symm_mem_helpers.py`, `SymmetricMemory` C++ |
| Sentinel stripping on write | codegen prologue |
| Remove epilogue sync | codegen epilogue |

The pragmatic near-term fix is gating the FX pass on tensor size: use
P2P with device-side sync for small tensors (where it wins), fall back
to NCCL for large tensors (where it doesn't).

## Persistent Reduction Analysis

The inductor-generated kernel forces persistent reduction via
`override_persistent_reduction = True` when the kernel contains
`symm_mem_p2p_reduce_load` ops. This is the correct choice:

- Without the override, the heuristic chooses looped reduction for
  `r0_numel=2880 > threshold=1024`, causing P2P loads to repeat per loop
  chunk: `world_size × num_chunks` NVLink loads instead of just
  `world_size`.
- The handwritten kernel (`fused_op`) also uses persistent-style reduction
  (`BLOCK_N = next_power_of_2(N)`, one block handles the full row).
- The performance gap at large token counts is caused by device-side sync
  scaling, not the reduction strategy. Evidence: `compiled` (49.8us) is
  nearly identical to `compiled_plain` (51.2us, standard NCCL + separate
  kernels) at 1024 tokens.

## GPU Execution Model for the Fused Kernel

Understanding how the generated kernel maps to GPU hardware is important
for reasoning about performance in the SGLang inference workload, where
token counts range from 1 (decode) to thousands (prefill).

### Thread hierarchy: CTAs, warps, threads

The generated kernel is a **persistent reduction**: each CTA (thread
block) loads the entire reduction dimension (`hidden_dim`) into registers
in one pass and reduces within shared memory.

For a tensor `[num_tokens, hidden_dim]` with `hidden_dim=2880`:
- `RBLOCK = next_power_of_2(2880) = 4096`
- Grid = `cdiv(num_tokens, XBLOCK)` CTAs
- Each CTA has `num_warps` warps (typically 4–8, chosen by autotuner)

```
4 warps (128 threads):  4096 / 128 = 32 elements per thread
8 warps (256 threads):  4096 / 256 = 16 elements per thread
```

The reduction (`pow → mean → rsqrt`) uses warp shuffles within each
warp (direct register exchange, ~1 cycle), then shared memory across
warps within the CTA (~tens of nanoseconds). All of this stays on a
single SM — no global memory coordination.

### SM utilization vs token count

| num_tokens | CTAs | SMs used (GB200, 152 SMs) | Utilization |
|-----------|------|--------------------------|-------------|
| 1 | 1 | 1 | 0.7% |
| 32 | 32 | 32 | 21% |
| 152 | 152 | 152 | 100% |
| 1024 | 1024 | 152 | 100%, ~7 CTAs/SM |

At `num_tokens = 1` (decode), only 1 SM does work — the kernel is fast
(~10-20us) but vastly underutilizes the GPU. This is inherent to
row-parallel reductions: a single output row must be reduced within a
single CTA because threads need shared memory to coordinate.

At `num_tokens >= 152`, all SMs are occupied and the kernel becomes
memory-bandwidth-bound (loading hidden_dim from each NVLink peer). This
is where host-barrier overhead (fixed ~6us for 2 barrier launches) is
amortized across many CTAs, and where two-shot allreduce (halving NVLink
traffic) provides the biggest win.

### Why persistent reduction is forced for P2P kernels

Inductor's default heuristic chooses looped reduction when
`rnumel > 1024` (for INNER reductions). With `hidden_dim = 2880`,
looped reduction would use `RBLOCK = 256` and loop 12 times. Each loop
iteration re-executes the P2P load from all peers — that's
`world_size × 12` NVLink loads instead of `world_size × 1`.

The kernel forces `override_persistent_reduction = True` to avoid this.
The cost is higher register pressure (`RBLOCK = 4096` means 32 elements
per thread with 4 warps), but NVLink latency (~microseconds per load)
dominates over register spill cost (~nanoseconds).

### Cooperative reduction (split-K)

For very large `rnumel` with few output rows, inductor can split the
reduction across multiple CTAs (cooperative reduction / split-K). Each
CTA reduces a chunk, then they combine partial results via a global
memory workspace.

This doesn't trigger for our workload: with `xnumel = 1` (decode),
the threshold is `32768 × 1 = 32768`, and `rnumel = 2880 < 32768`.
The coordination overhead (global memory writes + reads + an extra
kernel launch for the final reduce) isn't worth it when a single CTA
finishes the 2880-element reduction in ~2us.

## Patterns Matched

| Pattern | Matched | Notes |
|---------|---------|-------|
| `all_reduce → wait → rms_norm` | Yes | No residual, `pre_norm` output is `None` |
| `all_reduce → wait → add(wait, residual) → rms_norm` | Yes | With residual |
| `all_reduce → wait → add(residual, wait) → rms_norm` | Yes | Reversed add operand order |
| `all_reduce(avg) → wait → rms_norm` | Yes | Non-sum reduce ops |
| Multiple patterns in same graph | Yes | Each matched independently |
| `wait_tensor` has users outside pattern | No | Safety constraint — external consumers need the intermediate value |

## Requirements

The Triton kernel path requires symmetric memory (P2P access over NVLink).
Without P2P access, the feature still works — it just uses the decomposed
fallback (NCCL all_reduce + standard rms_norm).

## torch.compile Integration (SGLang-style)

```python
import torch.distributed._functional_collectives as funcol

@torch.compile(options={"_fused_all_reduce_rmsnorm": True})
def ar_norm(x, residual, weight, group_name, eps=1e-6):
    reduced = funcol.all_reduce(x, "sum", group_name)
    h = reduced + residual
    normed = F.rms_norm(h, weight.shape, weight, eps)
    return normed, h
```

The compiled function **must use `funcol.all_reduce()`** (not a custom wrapper)
because the FX pass matches `c10d_functional.all_reduce` nodes specifically.

**Setup requirements before first compile:**
1. `enable_symm_mem_for_group(group_name)` — registers the process group
2. `get_symm_mem_workspace(group_name, min_size=...)` — pre-allocates the
   P2P workspace (required before CUDA graph capture)
3. Run under `torch.inference_mode()` — required for the FX pass to fire

## Memory Pool Zero-Copy Variant

The default kernel path copies the input into a symmetric memory workspace
(`Memcpy DtoD`, ~5.7us for 16 MB bf16). The **memory pool variant** eliminates
this copy by allocating the upstream compute output directly in symmetric
memory.

**Setup (once, before CUDA graph capture):**
```python
import torch.distributed._symmetric_memory as symm_mem

mempool = symm_mem.get_mem_pool(device)
with torch.cuda.use_mem_pool(mempool):
    h_symm = torch.empty(M, N, device=device, dtype=torch.bfloat16)
sm_hdl = symm_mem.rendezvous(h_symm, dist.group.WORLD)
peer_bufs = _make_peer_bufs(sm_hdl, h_symm.shape, h_symm.dtype)
```

**Per-iteration (captured in CUDA graph):**
```python
torch.mm(intermediate, weight.t(), out=h_symm)   # write directly into symm mem
output, residual_out = _launch_fused_kernel(       # barrier → kernel → barrier
    sm_hdl, peer_bufs, h_symm, norm_weight, residual=residual, eps=eps,
)
```

## Profiling Results

### 2xGB200 wall-clock (HIDDEN=2880, CUDA graphs, --timer mode)

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

### 4xGB200 torch profiler (HIDDEN=2880, CUDA graphs)

NUM_TOKENS=32:

| Variant | Fused kernel (avg) | Total Self CUDA | Launches |
|---------|-------------------|-----------------|----------|
| baseline (NCCL) | 149us (NCCL) | 4.5ms | many |
| fused_op (host barriers) | 10.8us | 1.4ms | 3 |
| **compiled** (device sync) | 18.5us | **1.008ms** | **1** |
| **kraken** (device sync) | 15.1us | **1.002ms** | **1** |
| flashinfer (Lamport) | 9.2us | 837us | 1 |

## Test Coverage

Tests across 5 test classes:

| Category | Count | Description |
|----------|-------|-------------|
| Pattern matching | 8 | Positive/negative cases, multiple patterns, operand ordering |
| Op correctness (fallback) | 4 | Fallback impl numerical accuracy, return type semantics |
| Graph rewrite | 1 | Pass inserts fused op + getitem nodes |
| Helper functions | 2 | Node predicate helpers |
| Distributed (symm_mem) | 4 | Multi-GPU P2P kernel: no residual, with residual, per-rank data, 3D input |
| Distributed (fallback) | 4 | Multi-GPU fallback path via `_test_mode` |

## File Inventory

| File | Role |
|------|------|
| `torch/_inductor/config.py` | Config flag |
| `torch/_inductor/fx_passes/post_grad.py` | Wiring (passes `is_inference` to FX pass) |
| `torch/_inductor/fx_passes/fused_allreduce_rmsnorm.py` | FX pass (pattern match, replace, absorb trailing convert) |
| `torch/distributed/_symmetric_memory/_fused_all_reduce_rmsnorm.py` | Custom op (Meta/CUDA/fallback dispatch) |
| `torch/distributed/_symmetric_memory/_fused_allreduce_rmsnorm_triton.py` | Triton kernel (P2P loads, reduce, RMSNorm) |
| `torch/distributed/_symmetric_memory/__init__.py` | Op registration import |
| `test/distributed/test_fused_allreduce_rmsnorm.py` | Tests |
| `dev/profile_fused_allreduce_rmsnorm.py` | Profiling script (8 variants) |
| `third_party/kraken/` | Submodule: Triton-based symmetric memory operators |

## Reference: SGLang Custom AllReduce

SGLang's `tensor_model_parallel_all_reduce` dispatches to a custom CUDA
kernel (`sgl-kernel/csrc/allreduce/custom_all_reduce.cuh`) that beats
both NCCL and our Inductor P2P fused kernel for small messages.

Benchmark (`dev/benchmark.py`, 2xGB200, hidden=2880, seq_len=1):

| Variant | Time | Notes |
|---------|------|-------|
| SGLang custom AR + separate RMSNorm | 9 µs | 2+ kernels |
| Inductor P2P fused (allreduce+add+norm) | 11 µs | 1 kernel |
| funcol compiled (NCCL + compiled norm) | 17 µs | 2+ kernels |
| FlashInfer oneshot | 6 µs | 1 kernel |

### How it works

SGLang uses CUDA IPC handles (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`)
for direct NVLink P2P access between GPUs — conceptually the same as
PyTorch symmetric memory but at a lower level, bypassing the Python
`_SymmetricMemory` abstraction.

Two kernel variants are selected based on message size:
- **1-stage** (`cross_device_reduce_1stage`): all ranks read from all
  peers and reduce in one pass. Used for world_size=2 or small messages
  (< 512KB for ≤4 GPUs, < 256KB for ≤8 GPUs).
- **2-stage** (`cross_device_reduce_2stage`): reduce-scatter + allgather,
  halving NVLink traffic. Used for larger messages.

### Why it's faster than our compiled kernel at small sizes

1. **Native CUDA kernel, no Triton overhead.** The CUDA kernel uses
   128-bit aligned packed loads (`ld.128` / `st.128`) and hand-tuned
   PTX memory barriers. Triton adds overhead from its own code generation,
   register allocation, and block-level abstractions.

2. **Lightweight custom barrier.** Uses per-block counters with
   `st.release.sys.global` / `ld.acquire.sys.global` — just two
   NVLink atomics per block (similar to kraken CAS). But the grid is
   capped at 36 blocks regardless of token count: "too many SMs will
   cause contention on NVLink bus". This avoids the CAS scaling problem
   our kernel has at larger grids.

3. **No wrapper-level overhead.** No Python `symm_mem_setup` call, no
   cached dict lookup, no torch.ops dispatch. The kernel launch is a
   direct CUDA kernel call through a pre-initialized C++ object.

4. **No prologue copy.** The input is passed directly to the kernel
   which reads from it. No copy into a symmetric buffer — the IPC
   handles are set up once at initialization and the kernel reads
   directly from peers' allocations.

### Key difference: fixed grid cap vs. scaling grid

SGLang caps the grid at 36 blocks, then each block processes multiple
rows. This means the sync cost stays bounded: 36 × 2 barrier atomics =
72 NVLink atomics regardless of token count. Our compiled kernel
launches one block per XBLOCK rows, so the grid (and sync cost) grows
with token count.

This is the same insight behind the host-barrier approach: decouple sync
cost from grid size. SGLang achieves it by capping the grid; we can
achieve it by moving sync out of the kernel (host barriers) or by
capping XBLOCK to keep the grid small.

### Implications for our approach

The SGLang results show that for a serving runtime, a pre-compiled CUDA
kernel with fixed grid + direct IPC handles is hard to beat. Our Inductor
approach trades some performance for generality (works with any fused
downstream compute, auto-generated from user code). The key areas where
we can close the gap:

- **Cap the grid** or use host barriers to bound sync cost (host-barrier
  plan addresses this)
- **Eliminate prologue copy** via mempool zero-copy (already implemented
  as `_symm_mem_skip_prologue_copy`)
- **Reduce wrapper overhead** by moving `symm_mem_setup` to graph init

## Open Questions / Future Work

1. **Sync strategy selection.** Device-side per-block atomics don't scale
   beyond ~32 tokens. Options: (a) gate the FX pass on tensor size,
   (b) implement Lamport-style sentinel in codegen, (c) implement
   two-shot reduce-scatter + allgather for large tensors.

2. **Lamport-style sentinel sync for codegen.** Would keep single-launch
   advantage while scaling to large token counts. Requires push model,
   volatile loads, triple buffering, and sentinel stripping. See
   "Why We Can't Easily Adopt Lamport" section for full analysis.

3. **Two-shot P2P allreduce.** For large tensors, implement
   reduce-scatter + allgather to halve NVLink traffic from
   `world_size × data` to `2 × data` (matching FlashInfer's two-shot).

4. **LayerNorm variant.** Same pattern applies; kernel needs an additional
   mean-subtraction step.

5. **Training support.** AOTAutograd saves intermediates for backward that
   the fused op doesn't produce. Options: (a) return intermediates from
   fused op, (b) recompute in backward.

6. **Multi-node via NVSHMEM.** Replace `tl.load` with `nvshmem.get()`,
   `symm_mem.barrier()` with `nvshmem.barrier_all()`.

7. **Memory pool double buffering.** Eliminate epilogue sync by alternating
   two symmetric buffers across iterations.

8. **Non-sum reduce ops.** Supporting `avg` (divide by world_size after
   reduce) is trivial.

9. **Integration with compute-comm overlap.** Verify composition with
   `reorder_for_compute_comm_overlap` pass.
