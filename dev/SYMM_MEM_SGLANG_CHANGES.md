# Symmetric Memory Changes for SGLang Compiled MoE + AllReduce Fusion

Tracks all changes made to enable the Inductor P2P allreduce fusion pass
(`fuse_symm_mem_comms`) in SGLang's `--enable-torch-compile-moe` path.

## Problem

When `_fuse_symm_mem_comms: True` is set in `torch.compile` options, the
Inductor `fuse_symm_mem_comms_pass` should replace `all_reduce → wait_tensor`
with `symm_mem.p2p_allreduce`, fusing it with downstream RMSNorm into a single
Triton kernel. This was not happening in SGLang.

## Root Causes & Fixes

### 1. Pass gate: `is_symm_mem_enabled_for_group` returned False

**File:** `sglang/python/sglang/srt/model_executor/model_runner.py`

The Inductor pass checks `is_symm_mem_enabled_for_group(group_name)` which
requires `enable_symm_mem_for_group(group_name)` to have been called. SGLang
never called it.

**Fix:** Added `init_torch_symm_mem_for_compile()` in `ModelRunner.__init__`:

```python
def init_torch_symm_mem_for_compile(self):
    # gates: not draft, enable_torch_compile_moe, tp_size > 1
    _symm_mem.enable_symm_mem_for_group(tp_group_name)
    _symm_mem.get_symm_mem_workspace(tp_group_name, min_size=max_bs*hidden*2)
```

### 2. Timing: registration happened after compilation

**File:** `sglang/python/sglang/srt/model_executor/model_runner.py`

`init_torch_symm_mem_for_compile()` was initially placed after
`init_device_graphs()`, which triggers CUDA graph capture and `torch.compile`.
By the time the group was registered, the pass had already run and decided not
to replace.

**Fix:** Moved the call **before** the device init block:

```python
self.init_torch_symm_mem_for_compile()   # ← before compilation

if self.device == "cuda":
    self.init_cublas()
    self.init_attention_backend()
    self.kernel_warmup()
    self.init_device_graphs()             # ← torch.compile happens here
```

### 3. Workspace too small for varying batch sizes

**File:** `pytorch/torch/_inductor/runtime/symm_mem_helpers.py`

The runtime helper `_get_cached()` lazily allocates the P2P workspace on first
kernel call. If a later call arrives with a larger tensor (e.g. correctness test
uses 3 prompts while CUDA graphs were captured for batch size 1), the old code
hit a hard `assert`.

**Fix (PyTorch):** Changed the assert to evict-and-reallocate:

```python
if sm.buffer_size >= workspace_bytes:
    return cached
# Workspace too small — evict and re-allocate
del _symm_mem_cache[key]
# falls through to get_symm_mem_workspace(group_name, min_size=...)
```

This works outside CUDA graph capture. During capture, re-allocation is
impossible (`get_symm_mem_workspace` raises), which is why the SGLang
pre-allocation is also needed.

**Fix (SGLang):** Pre-allocate workspace sized for max CUDA graph batch size
before any graph capture:

```python
max_bs = self.server_args.cuda_graph_max_bs or 160
min_workspace = max_bs * hidden_size * 2  # bf16
_symm_mem.get_symm_mem_workspace(tp_group_name, min_size=min_workspace)
```

## Symmetric Memory Cache Architecture

Two-level cache, keyed by process group name:

```
L1: _group_name_to_workspace_tensor  (torch.distributed._symmetric_memory)
    - Owns the raw P2P tensor allocation
    - Populated by get_symm_mem_workspace()
    - Can grow (re-allocate + re-rendezvous) outside graph capture

L2: _symm_mem_cache  (torch._inductor.runtime.symm_mem_helpers)
    - Caches derived objects: (sm_handle, buf_ptrs, sig_ptrs, rank, world_size)
    - Populated on first kernel call by _get_cached()
    - Checks sm.buffer_size on hit; evicts if too small
```

Flow:

```
SGLang pre-alloc ──► seeds L1 with max-sized P2P tensor
                     (+ rendezvous, result discarded)
                         │
First kernel call ──► _get_cached() L2 miss
                      ──► get_symm_mem_workspace() hits L1 (no alloc)
                      ──► rendezvous (cheap, tensor already known)
                      ──► builds & caches (sm, ptrs...) in L2
                         │
Subsequent calls ───► L2 hit, buffer_size check passes → return immediately
```

## Generated Kernel (after all fixes)

Single fused Triton kernel doing:

1. **Prologue:** copy input → local symm_mem buffer
2. **Barrier:** `_symm_mem_sync` (kraken device-CAS)
3. **P2P loads:** accumulate from all peer buffers (allreduce sum in f32)
4. **Compute:** residual add + RMSNorm (fused, single pass)
5. **Epilogue:** `_symm_mem_sync` (signal reads complete)

## Future Optimizations

- **Skip prologue copy:** If MoE output lands directly in symm_mem pool
  (`_symm_mem_skip_prologue_copy`), eliminates the copy in step 1.
- **Two-shot allreduce:** `device_cas_2_shot` mode reduces NVLink traffic
  from `world_size × data_size` to `2 × data_size` for large batches.
- **Grid cap tuning:** `_symm_mem_grid_cap` controls CTA count for barrier
  cost vs parallelism tradeoff.

## Files Changed

| File | Change |
|------|--------|
| `sglang/.../model_runner.py` | `init_torch_symm_mem_for_compile()`: register group + pre-allocate workspace |
| `pytorch/.../symm_mem_helpers.py` | `_get_cached()`: evict-and-grow instead of assert |
