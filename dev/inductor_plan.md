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
| `dev/profile_fused_allreduce_rmsnorm.py` | 7-variant profiling script with --nsys mode |

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

Added FlashInfer `trtllm_allreduce_fusion` one-shot (no quant) as variant 7
in the profiling script for direct comparison.

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

## Improvements: Short Term

- [ ] Move `symm_mem_setup` to graph init (one-time, not per-call)
- [ ] Use `buffer_ptrs_dev` / `signal_pad_ptrs_dev` raw ints
- [ ] Fix bf16 hardcoding → use actual input dtype
- [ ] Profile on realistic shapes (hidden=4096/8192, seq=2048)

## Improvements: Medium Term

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
