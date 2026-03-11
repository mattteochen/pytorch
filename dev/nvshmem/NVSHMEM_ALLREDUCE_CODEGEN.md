# NVSHMEM Allreduce Codegen for PyTorch Inductor

**Authors:** (authored with Claude)
**Status:** Codegen and compilation working; signal address issue under investigation
**Oncall:** distributed

## Overview

This adds a fifth sync mode `"nvshmem"` to the inductor symmetric-memory P2P
allreduce codegen. The existing four modes (`host_barrier`, `device_cas`,
`device_cas_2_shot`, `lamport`) all rely on NVLink-mapped peer buffers with
host-side barriers or kraken's device-side CAS/Lamport synchronization.

The NVSHMEM mode uses NVSHMEM signal primitives (`signal_op` +
`signal_wait_until`) for device-side synchronization. Unlike the other
device-side modes, NVSHMEM works across both NVLink (intra-node) and
RDMA/InfiniBand (inter-node), enabling multi-node fused allreduce+compute
kernels.

## Architecture

The implementation reuses the entire existing pipeline (FX pass, custom op, IR
lowering, IR node) unchanged. Changes are confined to codegen, runtime helpers,
and the compilation layer.

### Signal Protocol

Pull-model allreduce using monotonic epoch-based signals:

1. Copy input to local symmetric memory buffer
2. `fence()` — ensure stores are globally visible
3. Signal each peer that data is ready (prologue signal slots)
4. Wait for each peer's data-ready signal
5. Accumulate from all peer buffers
6. (epilogue) Signal reads complete, wait for peers' read-complete signals

Signal layout per rank: `2 * world_size` uint64 slots in the signal pad.
Slots `[0..world_size-1]` are prologue (data-ready) signals; slots
`[world_size..2*world_size-1]` are epilogue (reads-done) signals. The epoch
increments each invocation, so signals never need resetting — critical for
CUDA graph replay.

## Files Changed

### Codegen

| File | Change |
|------|--------|
| `torch/_inductor/config.py` | Added `"nvshmem"` to `_symm_mem_sync_mode` doc |
| `torch/_inductor/codegen/triton_symm_mem.py` | State init, body codegen (`_codegen_nvshmem_reduce_load`), prologue/epilogue (`codegen_nvshmem_prologue/epilogue`), wrapper setup (`emit_nvshmem_setup`) |
| `torch/_inductor/codegen/triton.py` | Wired NVSHMEM into 7 integration points: sync mode eval, import injection, argdefs (`_nvshmem_epoch`), `triton_meta` (`extern_libs`), `inductor_meta` (`nvshmem_init`), prologue/epilogue dispatch, `call_kernel` dispatch. Added grid-cap exclusion guards. |

### Runtime

| File | Change |
|------|--------|
| `torch/_inductor/runtime/nvshmem_helpers.py` | **New file.** Provides `nvshmem_peer_bufs()` (returns peer buffer views + signal pad pointers), `nvshmem_get_epoch()` (monotonic epoch counter), and `set_nvshmem_workspace()` for pre-registering the NVSHMEM workspace handle. |
| `torch/_inductor/runtime/triton_heuristics.py` | Passes `extern_libs` into `CUDAOptions` (the `options` dict for `triton.compile`). Calls `_nvshmemx_cumodule_init()` on the compiled binary's CU module when `nvshmem_init` is set. |

### Tests

| File | Change |
|------|--------|
| `test/distributed/test_fuse_symm_mem_comms.py` | Added `_assert_nvshmem_codegen` (verifies signal ops present, CAS/Lamport/host_barrier absent), `test_torch_compile_nvshmem_allreduce_sum`, `test_torch_compile_nvshmem_upstream_allreduce_sum`. |

### Unchanged (Reused As-Is)

- `torch/_inductor/fx_passes/fuse_symm_mem_comms.py` — FX pass is sync-mode agnostic
- `torch/distributed/_symmetric_memory/_p2p_allreduce.py` — custom op unchanged
- `torch/_inductor/comm_lowering.py` — IR lowering unchanged
- `torch/_inductor/ir.py` (`SymmMemP2PAllReduce`) — IR node unchanged
- `torch/distributed/_symmetric_memory/_nvshmem_triton.py` — NVSHMEM Triton wrappers (imported by generated code)

## Integration Points in `triton.py`

The NVSHMEM mode adds a `_symm_mem_use_nvshmem` boolean flag alongside the
existing `_symm_mem_use_host_barriers` and `_symm_mem_use_lamport`. The flag
is checked at seven locations in `codegen_kernel` and `call_kernel`:

1. **Sync mode re-evaluation** — sets the three mode flags correctly
2. **Import injection** — emits `from _nvshmem_triton import fence, signal_op, signal_wait_until`
3. **Kernel argdefs** — adds `_nvshmem_epoch` as a `SizeArg` (non-constexpr int)
4. **`triton_meta["extern_libs"]`** — links `libnvshmem_device.bc` via `NvshmemLibFinder`
5. **`inductor_meta["nvshmem_init"]`** — signals `triton_heuristics.py` to init the CU module
6. **Prologue/epilogue dispatch** — routes to `codegen_nvshmem_prologue/epilogue`
7. **`call_kernel` wrapper setup** — routes to `emit_nvshmem_setup`

Grid-cap guards also exclude NVSHMEM (grid-stride loops are device_cas-only).

## Compilation Flow

NVSHMEM kernels require linking against `libnvshmem_device.bc` (bitcode) at
Triton compile time, and initializing the resulting CU module with NVSHMEM's
runtime. The flow:

```
codegen (triton.py)
  → triton_meta["extern_libs"] = {"libnvshmem_device": "/path/to/libnvshmem_device.bc"}
  → inductor_meta["nvshmem_init"] = True

triton_heuristics.py (_precompile_config)
  → options["extern_libs"] = compile_meta["extern_libs"]   # goes into CUDAOptions
  → binary = triton.compile(src, target, options)
  → binary.run                                              # force module loading
  → _nvshmemx_cumodule_init(binary.module)
```

Note: `extern_libs` is a field on Triton's `CUDAOptions` dataclass, not a
top-level kwarg to `triton.compile()`.

## NVSHMEM Workspace Pre-allocation

Unlike other sync modes that use `get_symm_mem_workspace()` (which passes
`group_name` to `empty_strided_p2p`), the NVSHMEM allocator rejects
`group_name`. NVSHMEM allocation is also a collective operation requiring all
ranks to participate simultaneously, so it cannot happen lazily inside the
inductor-generated wrapper code.

The solution: pre-allocate the workspace before `torch.compile`:

```python
import torch.distributed._symmetric_memory as symm_mem
from torch._inductor.runtime.nvshmem_helpers import set_nvshmem_workspace

symm_mem.set_backend("NVSHMEM")
ws_tensor = symm_mem.empty(size_bytes, dtype=torch.uint8, device=device)
sm_handle = symm_mem.rendezvous(ws_tensor, group=group_name)
set_nvshmem_workspace(group_name, sm_handle)
```

The runtime helper `nvshmem_peer_bufs()` then looks up the pre-registered
workspace via `_nvshmem_workspace[group_name]`.

## Triton Type Casting

NVSHMEM extern functions expect specific Triton types:

- `signal_op(int64, int64, int32, int32)` — address, value, op, pe
- `signal_wait_until(int64, int32, uint64)` (via JIT wrapper that casts internally)
- `fence()` — no args

Constexpr int expressions (e.g. `SYMM_RANK * 8`) and scalar kernel args
(e.g. `_nvshmem_epoch`) don't support `.to()` in Triton. Use `tl.cast(expr,
tl.int64)` instead. The `signal_wait_until` JIT wrapper handles `cmp_val`
casting internally, so pass `_nvshmem_epoch` directly without casting.

## Usage

```python
import torch.distributed._symmetric_memory as symm_mem
from torch._inductor.runtime.nvshmem_helpers import set_nvshmem_workspace

# 1. Set up NVSHMEM workspace (collective, before compile)
symm_mem.set_backend("NVSHMEM")
ws = symm_mem.empty(workspace_bytes, dtype=torch.uint8, device=device)
sm = symm_mem.rendezvous(ws, group=group_name)
set_nvshmem_workspace(group_name, sm)

# 2. Compile with NVSHMEM sync mode
@torch.compile(options={
    "_fuse_symm_mem_comms": True,
    "_symm_mem_sync_mode": "nvshmem",
})
def fn(x, residual, group_name):
    reduced = all_reduce(x, "sum", group=group_name)
    h = reduced + residual
    return rms_norm(h, weight, eps)
```

## Current Status

**Working end-to-end:**
- FX pass fires and replaces `all_reduce + wait` with `p2p_allreduce`
- Codegen produces correct NVSHMEM Triton kernel with signal ops
- Triton compilation succeeds with `extern_libs` linking `libnvshmem_device.bc`
- `_nvshmemx_cumodule_init` called on the compiled CU module
- Runtime workspace pre-allocation via `set_nvshmem_workspace()`
- Kernel launches on GPU

**Open issue — signal address computation:**

The kernel hits `cudaErrorIllegalAddress` during NVSHMEM signal operations.
The `signal_pad_ptrs` from `NVSHMEMSymmetricMemory` are derived via
`nvshmem_ptr(signal_pad_ptr, remote_rank)`, which returns local-mapped
addresses of remote signal pads. The open question is whether
`nvshmemx_signal_op(addr, ...)` expects:

- (a) The **symmetric** base address (same virtual address on all PEs), or
- (b) The **local-mapped** address from `nvshmem_ptr()`

The existing `_nvshmem_triton.py` tests use `get_signal_pad(rank, shape,
dtype)` which returns a tensor view — its `.data_ptr()` is a symmetric
address. The `signal_pad_ptrs` vector from `NVSHMEMSymmetricMemory` may use a
different address space. This needs investigation to determine the correct
address derivation for the inductor codegen path.

## Verification

```bash
# Unit tests
python test/distributed/test_fuse_symm_mem_comms.py -k nvshmem

# Generated code inspection
TORCH_LOGS=output_code torchrun --nproc_per_node=4 your_script.py

# Profiling
torchrun --nproc_per_node=4 dev/profile_fused_allreduce_rmsnorm.py --timer --num-tokens 1
```

## Issues Encountered During Implementation

1. **`triton.compile()` does not accept `extern_libs` as a kwarg** — it must
   go into the `options` dict (becomes `CUDAOptions.extern_libs`).

2. **Triton type errors with `.to()` on ints** — constexpr products like
   `SYMM_RANK * 8` and scalar args like `_nvshmem_epoch` are plain ints in
   Triton, not tensors. Use `tl.cast()` for address arithmetic; pass epoch
   directly to `signal_wait_until` (the JIT wrapper casts internally).

3. **`signal_op` type dispatch** — the extern expects `(int64, int64, int32,
   int32)`. Literal `1` infers as `int32`, so the signal value must be
   explicitly cast: `tl.cast(1, tl.int64)`.

4. **NVSHMEM allocator rejects `group_name`** — `get_symm_mem_workspace()`
   passes `group_name` to `empty_strided_p2p`, which the NVSHMEM allocator
   does not support. Workspace must be pre-allocated via `symm_mem.empty()` +
   `rendezvous()`.

5. **NVSHMEM init is collective** — `set_backend("NVSHMEM")` +
   `symm_mem.empty()` triggers `nvshmemx_init_attr()` which all ranks must
   call together. This cannot happen inside the inductor wrapper (runs
   per-rank asynchronously). Solution: pre-allocate before `torch.compile`.
