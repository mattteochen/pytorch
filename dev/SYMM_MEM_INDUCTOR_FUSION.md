# Symmetric Memory P2P AllReduce + RMSNorm Fusion in Inductor

Tracks the work to enable Inductor's P2P allreduce fusion pass
(`fuse_symm_mem_comms`) for SGLang's compiled MoE decode path.

## What it does

When `_fuse_symm_mem_comms: True` is set in `torch.compile` options, Inductor
replaces `all_reduce → wait_tensor` with `symm_mem.p2p_allreduce`. The
scheduler then fuses the P2P loads with downstream compute (residual add +
RMSNorm) into a **single Triton kernel** — eliminating the NCCL allreduce and
inter-kernel launch overhead.

### Before (NCCL path)
```
kernel 1: copy to buffer
NCCL all_reduce_ (ring/tree, host-initiated)
wait_tensor
kernel 2: residual add + RMSNorm
```

### After (P2P fused path)
```
single kernel:
  prologue: copy input → local symm_mem buffer
  barrier:  kraken CAS sync
  P2P load: accumulate from peer buffers [0..world_size-1]
  compute:  residual add + RMSNorm
  epilogue: kraken CAS sync (signal reads complete)
```

## Changes made

### PyTorch (`torch/_inductor/runtime/symm_mem_helpers.py`)

**`_get_cached` — graceful workspace growth instead of assert**

The old code asserted if the cached symmetric memory workspace was too small
for a new input. This crashed when batch sizes grew (e.g. correctness test
uses 3 prompts but CUDA graphs were captured for batch size 1).

Fix: evict the cache entry and re-allocate via `get_symm_mem_workspace`,
which grows the underlying P2P tensor. This works outside graph capture;
during graph capture the pre-allocation (see below) ensures it never triggers.

### SGLang (`sglang/srt/model_executor/model_runner.py`)

**`init_torch_symm_mem_for_compile` — new method in ModelRunner**

Called before CUDA graph capture / torch.compile. Does two things:

1. **`enable_symm_mem_for_group(tp_group_name)`** — registers the TP group in
   PyTorch's `_group_name_to_store` dict so `is_symm_mem_enabled_for_group()`
   returns `True`. Without this, the Inductor pass skips the replacement.

2. **`get_symm_mem_workspace(tp_group_name, min_size=...)`** — pre-allocates
   the P2P workspace sized for `cuda_graph_max_bs * hidden_size * 2` (bf16).
   This ensures no re-allocation is ever needed during CUDA graph capture.

**Placement**: must run *before* `init_device_graphs()` which triggers
`torch.compile` via CUDA graph capture warmup. Previously placed after — the
pass had already run and decided not to replace.

## Symmetric memory cache architecture

Two-level cache, keyed by process group name:

```
┌─────────────────────────────────────────────────────────────┐
│  L1: _group_name_to_workspace_tensor  (torch.distributed)   │
│  Owns the raw P2P tensor allocation.                        │
│  Populated by: get_symm_mem_workspace()                     │
│  Grows by: re-allocating + re-rendezvous                    │
│  Cannot grow during CUDA graph capture.                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ first kernel call
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  L2: _symm_mem_cache  (torch._inductor.runtime)             │
│  Caches derived objects: sm handle, buf_ptrs, sig_ptrs,     │
│  rank, world_size.                                          │
│  Populated by: _get_cached() on first compiled kernel call  │
│  Evicts when: sm.buffer_size < input size (our fix)         │
└─────────────────────────────────────────────────────────────┘
```

**Init flow:**
```
SGLang pre-alloc ──► seeds L1 with max-sized P2P tensor
                     (+ rendezvous, result discarded)
                           │
first compiled kernel ─────► L2 cache miss
                             calls get_symm_mem_workspace()
                             L1 hit (tensor large enough, no alloc)
                             re-rendezvous (cheap, idempotent)
                             caches (sm, buf_ptrs, sig_ptrs, ...) in L2
                           │
subsequent calls ──────────► L2 cache hit → return immediately
```

## Key config options

| Option | Default | Effect |
|--------|---------|--------|
| `_fuse_symm_mem_comms` | `False` | Enable the FX pass |
| `_fuse_symm_mem_comms_max_bytes` | 1MB | Size gate; 0 = no limit |
| `_symm_mem_sync_mode` | `"host_barrier"` | Sync protocol: `host_barrier`, `device_cas`, `device_cas_2_shot`, `lamport` |
| `_symm_mem_grid_cap` | None | Max CTAs for barrier (tune for decode batch sizes) |
| `_symm_mem_skip_prologue_copy` | `False` | Skip copy if input is already in symm_mem (mempool path) |

## Potential further optimizations

1. **Skip prologue copy**: if the MoE second `grouped_mm` output lands
   directly in symm_mem (via `torch.cuda.use_mem_pool`), the copy from
   regular CUDA memory → symm_mem buffer is eliminated.

2. **Two-shot allreduce** (`device_cas_2_shot`): each peer reduces a shard
   then broadcasts. Reduces NVLink traffic from `world_size * data_size` to
   `2 * data_size`. Matters for larger batch sizes.

3. **GDC barrier removal**: the fused kernel has `gdc_wait` /
   `gdc_launch_dependents` from `_symm_mem_sync_mode: "device_cas"` in
   addition to the kraken `_symm_mem_sync` barriers. If all kernels run on
   the same CUDA stream, the GDC pair may be redundant.

4. **Single-level cache**: merge L1 and L2 into one cache that atomically
   manages the P2P tensor, sm handle, and pointer tensors together.

## Files involved

- `torch/_inductor/fx_passes/fuse_symm_mem_comms.py` — the FX pass
- `torch/_inductor/fx_passes/post_grad.py` — pass invocation (gated on config)
- `torch/_inductor/runtime/symm_mem_helpers.py` — runtime cache + setup helpers
- `torch/distributed/_symmetric_memory/__init__.py` — P2P allocation + rendezvous
- `sglang/srt/model_executor/model_runner.py` — SGLang init hook
- `sglang/srt/models/gpt_oss.py` — `_compiled_moe_norm` compile options
