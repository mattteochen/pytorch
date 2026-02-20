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

**Host-side flow:**
```
1. get_symm_mem_workspace(group_name, input_bytes)  — cached allocation
2. Copy local input into the symmetric buffer
3. symm_mem.barrier(channel=0) — all ranks' data is now visible
4. Launch Triton kernel — P2P loads from all ranks' buffers
5. symm_mem.barrier(channel=0) — safe to reuse buffers
```

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

**Synchronization:** Host-side `symm_mem.barrier()` provides lightweight
synchronization before and after the kernel. This is the same mechanism used
by all existing symmetric memory ops in PyTorch (`_pipelined_all_gather`,
`_fused_all_gather_matmul`, etc.). No NVSHMEM dependency required.

**GPU timeline (3 kernel launches per iteration):**
```
barrier (~5us) → fused kernel (P2P loads + reduce + RMSNorm) → barrier (~5us)
```

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

The Triton kernel path requires symmetric memory (P2P access over NVLink):

```
                             ┌─────────────────────────┐
  NVLink-connected GPUs ─────┤  get_symm_mem_workspace  │
  CUDA backend ──────────────┤  symm_mem.barrier()      │
  Triton ────────────────────┤  P2P tl.load from peers  │
                             └─────────────────────────┘
```

Without P2P access, the feature still works — it just uses the decomposed
fallback (NCCL all_reduce + standard rms_norm).

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
| `dev/profile_fused_allreduce_rmsnorm.py` | Profiling script (4 variants: baseline, fused_op, compiled, mempool) |

## torch.compile Integration (SGLang-style)

The intended integration path for serving frameworks like SGLang:

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

A profiling script (`dev/profile_fused_allreduce_rmsnorm.py`) demonstrates the
full integration with a dummy decoder layer (linear → allreduce → add residual
→ RMSNorm), comparing four variants under CUDA graph capture: NCCL baseline,
direct fused op (workspace copy), torch.compile (FX pass fusion), and memory
pool zero-copy (matmul output in symmetric memory).

## Profiling Results (4× GPU, H100, hidden=4096, seq=2048)

| Variant | CUDA total | Key kernels |
|---------|-----------|-------------|
| **Baseline** (NCCL) | ~30.8ms | NCCL allreduce 24.8ms, mm 3.8ms, rms_norm 1.7ms |
| **Fused op** (direct) | ~17.0ms | barrier 11.1ms, fused kernel 1.5ms, mm 3.9ms |
| **Compiled** (torch.compile) | Similar to fused op | FX pass fires: "found 1 patterns / fused 1 patterns" |

The fused Triton kernel replaces NCCL allreduce (24.8ms) + add + rms_norm
(1.7ms) with a single 1.5ms kernel, at the cost of two barrier kernels
(~11ms total in this config — dominated by barrier wait time, not overhead).

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

**Key details:**

- `rendezvous` is a collective on first call (IPC handle exchange) but cached
  per block+group thereafter — effectively free in steady state.
- `use_mem_pool` and `rendezvous` happen *before* CUDA graph capture so the
  graph only sees GPU ops (matmul, barriers, Triton kernel) with fixed
  addresses.
- The pre-kernel barrier is required (wait for all ranks' matmul to finish).
  The post-kernel barrier is also required in a loop — without it, a rank
  could overwrite its symmetric buffer (via the next iteration's matmul) while
  another rank is still reading from it. Eliminating the post-barrier would
  require double buffering.

**Profiling results (4× GPU, CUDA graphs, hidden=4096, seq=2048):**

| Metric | fused_op (workspace copy) | mempool (zero-copy) |
|--------|--------------------------|---------------------|
| Memcpy DtoD | 114us (5.7us/call) | **0** |
| barrier (40 calls) | 723us (18.1us/call) | 944us (23.6us/call) |
| fused kernel (20 calls) | 1.657ms (82.8us/call) | 1.682ms (84.1us/call) |
| **Total CUDA** | **6.682ms** | **6.802ms** |

The DtoD copy is eliminated but the total is ~1.8% slower due to slightly
higher barrier cost (likely TLB/cache effects from the different backing
allocation). For this tensor size (16 MB) the copy is only ~5.7us — the
optimization becomes more relevant for larger tensors or when composing with
upstream kernels that can also write directly to the pool.

## Open Questions / Future Work

1. **Device-side synchronization.** The current implementation uses host-side
   `symm_mem.barrier()` before and after the kernel launch. Moving to
   device-side synchronization (porting `sync_remote_blocks` signal pad
   protocol to Triton, or using kraken's `ptx_utils.symm_mem_sync`) would
   eliminate the two barrier kernel launches (~10us total). Blocked on Triton
   exposing memory ordering semantics for system-scope atomics.

2. **LayerNorm variant.** The same pattern applies to `all_reduce + LayerNorm`.
   The kernel would need an additional mean-subtraction step. The FX pass
   would need to match the LayerNorm decomposition (which includes a mean
   subtraction before variance computation).

3. **Non-sum reduce ops.** The Triton kernel currently only supports `sum`.
   Supporting `avg` requires dividing by world_size after the reduction, which
   is trivial to add.

4. **Performance tuning.** The kernel uses `BLOCK_N = next_power_of_2(N)` which
   may not be optimal for all hidden dimensions. Auto-tuning or a lookup table
   based on common hidden sizes (4096, 5120, 8192, etc.) could improve
   throughput.

5. **Integration with compute-comm overlap.** The fused op changes the
   scheduling picture for `reorder_for_compute_comm_overlap` since the
   communication is now inside the fused kernel rather than a separate NCCL
   call. Need to verify the two passes compose correctly.

6. **Multi-node support.** The current kernel uses direct P2P loads which
   require NVLink (intra-node). For multi-node TP, the P2P loads could be
   replaced with `nvshmem.get()` and `symm_mem.barrier()` with
   `nvshmem.barrier_all()`, at the cost of requiring NVSHMEM and cooperative
   launch.

7. **Training support.** The FX pass currently skips training mode because
   AOTAutograd saves intermediates (e.g. `rsqrt`) for backward that the fused
   op doesn't produce. Supporting training would require either: (a) having the
   fused op also return the intermediates, or (b) recomputing them from the
   pre_norm output in the backward graph.

8. **Memory pool double buffering.** The post-kernel barrier (~5us) could be
   eliminated by alternating between two symmetric memory buffers across
   iterations. While rank N reads from buffer A, the next matmul writes to
   buffer B — removing the read-after-write hazard without synchronization.
