# Inductor-Generated Fused AllReduce via Kraken PTX

## Status: Working E2E on 4xGB200

All 22 tests pass (14 single-process + 6 multi-GPU distributed + 2 torch.compile e2e).
The `torch.compile` path generates a single fused Triton kernel that does
P2P allreduce + residual add + RMSNorm with kraken device-side sync.

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
| `torch/_inductor/codegen/triton.py` | P2P load codegen, prologue/epilogue, kraken import, constants, call_kernel |
| `torch/_inductor/codegen/triton_utils.py` | `is_unaligned_buffer` safety for virtual buffers |
| `torch/_inductor/ir.py` | `SymmMemP2PAllReduce` factory (creates fusible Pointwise) |
| `torch/_inductor/fx_passes/fused_allreduce_rmsnorm.py` | Simplified FX pass (only replaces all_reduce+wait) |
| `torch/_inductor/comm_lowering.py` | Lowering for `symm_mem.p2p_allreduce` |
| `torch/_inductor/dtype_propagation.py` | Dtype rule for `symm_mem_p2p_reduce_load` |
| `torch/_inductor/dependencies.py` | Read-dep tracking |
| `torch/_inductor/loop_body.py` | Index forwarding |
| `torch/_inductor/sizevars.py` | Index simplification forwarding |
| `torch/distributed/_symmetric_memory/_p2p_allreduce.py` | Custom op (Meta + fallback) |
| `torch/distributed/_symmetric_memory/__init__.py` | Import registration |
| `torch/_inductor/runtime/symm_mem_helpers.py` | Wrapper runtime: cached workspace + pointer tensors |
| `test/distributed/test_fused_allreduce_rmsnorm.py` | 22 tests (pattern, op, compile e2e, multi-GPU) |

## Generated Kernel vs Kraken Reference

| Aspect | Generated | Kraken handwritten |
|--------|-----------|-------------------|
| Copy-in | `tl.load(buffer_ptrs + RANK)` → `tl.store` | Same |
| Sync | `_symm_mem_sync(signal_pad_ptrs, ...)` | Same function |
| P2P reduce | `tl.static_range(WS)` loop, pointer-to-pointer deref | `range(world_size)` loop, same deref |
| RMSNorm | **Inductor-generated** (pow→mean→rsqrt→mul) | Hand-coded |
| Grid | `xnumel` programs (XBLOCK=1 forced) | `num_blocks` (one per row) |
| Kernel launches | 1 | 1 |

## Current Overhead

### 1. `symm_mem_setup` called per kernel invocation

`symm_mem_helpers.symm_mem_setup()` is called from the wrapper before
every kernel launch.  It is cached (dict lookup), but still runs Python
per call:

```python
# Generated wrapper (output_code.py line 164-165):
_symm_buf_ptrs, _symm_signal_pad_ptrs, _symm_rank, _symm_world_size = symm_mem_setup(arg0_1, "0")
triton_per_fused_...run(arg0_1, arg1_1, arg2_1, buf0, buf2, 4, 64, _symm_buf_ptrs, _symm_signal_pad_ptrs, ...)
```

**Fix:** Move to `Runner.__init__` (one-time) or to module-level init.
The pointer tensors are stable after rendezvous.

### 2. Two CUDA int64 tensors allocated for pointer arrays

`symm_mem_setup` creates `torch.tensor([ptr0, ptr1, ...], device=cuda)`
for `buffer_ptrs` and `signal_pad_ptrs`.  Small (world_size elements)
but real CUDA allocations on first call.

**Fix:** Use `SymmetricMemory.buffer_ptrs_dev` / `signal_pad_ptrs_dev`
directly.  These are raw ints pointing to device-resident arrays
maintained by the C++ runtime — no Python tensor wrapper needed.
Requires solving the `TensorArg` vs raw-int issue in
`triton_heuristics` (the heuristics wrapper expects typed args in
`triton_meta.signature`).

### 3. XBLOCK forced to 1

Required for grid consistency (all ranks must launch the same number
of programs for kraken's per-block sync).  Prevents inductor from
using larger XBLOCK for better occupancy.

**Fix:** Allow XBLOCK > 1 by adjusting the prologue to copy XBLOCK
rows per program.  Grid = `cdiv(xnumel, XBLOCK)` is consistent
across ranks if XBLOCK and xnumel are the same.  Risk: autotuning
could pick different XBLOCK per rank.  Mitigate by broadcasting the
autotuning result from rank 0.

### 4. bf16 hardcoded in prologue pointer cast

```python
_symm_local_buf = tl.load(_symm_bptrs + SYMM_RANK).to(tl.pointer_type(tl.bfloat16))
```

**Fix:** Use the actual input dtype from `V.graph.get_dtype(name)`.

### 5. Redundant `symm_buf_ptrs.to(pointer_type)` cast

Done once in the prologue and once in the P2P load body.
Minor — Triton optimizes it away.

## Improvements: Short Term

- [ ] Move `symm_mem_setup` to graph init (one-time, not per-call)
- [ ] Use `buffer_ptrs_dev` / `signal_pad_ptrs_dev` raw ints instead
      of wrapping in `torch.tensor`
- [ ] Fix bf16 hardcoding → use actual input dtype
- [ ] Profile the generated kernel vs kraken reference on realistic
      shapes (hidden=4096/8192, seq=2048)
- [ ] Clean up: remove `symm_buf_ptr` arg (no longer used, prologue
      loads from `buffer_ptrs + RANK`)

## Improvements: Medium Term

- [ ] **LayerNorm variant:** The FX pass only replaces `all_reduce + wait`;
      any downstream compute fuses automatically.  LayerNorm = same
      pipeline, inductor already generates LayerNorm kernels.
- [ ] **Training support:** Currently inference-only (FX pass gated on
      `is_inference`).  Training requires the fused kernel to produce
      intermediates for autograd (e.g. rsqrt for backward).  Approach:
      extra Pointwise outputs, or recomputation in backward.
- [ ] **Memory pool zero-copy:** Allocate upstream compute output
      directly in symmetric memory via `symm_mem.get_mem_pool()`,
      eliminating the prologue copy entirely.
- [ ] **XBLOCK > 1 with deterministic autotuning:** Broadcast autotuning
      config from rank 0.  Larger XBLOCK → better occupancy for many rows.
- [ ] **Non-sum reduce ops:** `avg` needs `/ world_size` after reduce.
      Trivial addition to the P2P load codegen.

## Improvements: Long Term

- [ ] **Generalize beyond allreduce:** The P2P load + sync pattern applies
      to reduce_scatter, all_gather.  Parameterize the sync pattern and
      P2P access.
- [ ] **Multi-node via NVSHMEM:** Replace `tl.load(peer_buf + offset)` with
      `nvshmem.get()`.  Kernel structure stays the same.
- [ ] **Double buffering:** Alternate between two symmetric memory buffers
      across iterations to eliminate the epilogue sync.
- [ ] **CUDA graph capture:** The current design is CUDA-graph friendly
      (kraken's signal pad auto-resets, pointers are stable).  Needs
      testing under `torch.cuda.graph()` capture.
- [ ] **Upstream to PyTorch:** The new `ops.symm_mem_p2p_reduce_load`
      primitive, `SymmMemP2PAllReduce` IR node, and codegen changes
      should be upstreamed as a first-class inductor feature for
      P2P communication fusion.
