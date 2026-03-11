# NVSHMEM Allreduce Codegen for PyTorch Inductor

**Authors:** (authored with Claude)
**Status:** Implementation complete
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
| `torch/_inductor/runtime/nvshmem_helpers.py` | **New file.** Mirrors `symm_mem_helpers.py`. Provides `nvshmem_peer_bufs()` (returns peer buffer views + signal pad pointers) and `nvshmem_get_epoch()` (monotonic epoch counter). Uses `get_symm_mem_workspace()` for buffer allocation. |
| `torch/_inductor/runtime/triton_heuristics.py` | Passes `extern_libs` through to `triton.compile()` kwargs. Calls `_nvshmemx_cumodule_init()` on the compiled binary's CU module when `nvshmem_init` is set. |

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
  → compile_kwargs["extern_libs"] = compile_meta["extern_libs"]
  → binary = triton.compile(..., extern_libs=...)
  → _nvshmemx_cumodule_init(binary.module)
```

## Usage

```python
@torch.compile(options={
    "_fuse_symm_mem_comms": True,
    "_symm_mem_sync_mode": "nvshmem",
})
def fn(x, residual, group_name):
    reduced = all_reduce(x, "sum", group=group_name)
    h = reduced + residual
    return rms_norm(h, weight, eps)
```

Prerequisite: the user must have NVSHMEM installed and the symmetric memory
backend set to NVSHMEM (`symm_mem.set_backend("NVSHMEM")`).

## Verification

```bash
# Unit tests (codegen correctness, no NVSHMEM hardware needed for codegen tests)
python test/distributed/test_fuse_symm_mem_comms.py -k nvshmem

# Generated code inspection
TORCH_LOGS=output_code torchrun --nproc_per_node=4 your_script.py

# Profiling
torchrun --nproc_per_node=4 dev/profile_fused_allreduce_rmsnorm.py --timer --num-tokens 1
```
