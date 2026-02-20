# Integration Guide: Fused AllReduce + RMSNorm for SGLang

This document provides an AI agent with all context needed to integrate
PyTorch's fused allreduce + RMSNorm into the SGLang serving framework.

## What This Feature Does

In tensor-parallel inference, every decoder layer runs:

```
allreduce(x) → x + residual → rms_norm(x)
```

This is normally 3+ kernel launches (NCCL allreduce, pointwise add, RMSNorm).
The fused version replaces them with a **single Triton kernel** that performs
P2P loads from all ranks via NVLink symmetric memory, reduces, adds residual,
and normalizes — all in one pass without materializing intermediates.

## Prerequisites

- PyTorch with symmetric memory support (NVLink-connected GPUs, single node)
- The feature lives in `torch.distributed._symmetric_memory` and
  `torch._inductor.fx_passes.fused_allreduce_rmsnorm`

## Integration Path: torch.compile (Recommended)

This is the cleanest path. The model code uses standard `funcol.all_reduce` +
`F.rms_norm`, and `torch.compile` with an FX pass automatically fuses them.

### Step 1: Identify the allreduce + norm pattern in the model

Find where the model does something like:

```python
# Typical pattern after MoE/FFN down_proj:
hidden_states = all_reduce(hidden_states)           # NCCL collective
hidden_states = hidden_states + residual            # residual add
hidden_states = rms_norm(hidden_states, weight)     # normalization
residual = hidden_states                            # save for next layer
```

### Step 2: Extract the allreduce + norm into a compilable function

The allreduce and norm must be inside the same `torch.compile` region.
**Critical:** use `torch.distributed._functional_collectives.all_reduce`
(not `dist.all_reduce` or any custom wrapper) because the FX pass matches
`c10d_functional.all_reduce` nodes specifically.

```python
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F

@torch.compile(options={"_fused_all_reduce_rmsnorm": True})
def fused_ar_norm(x, residual, weight, group_name, eps=1e-5):
    reduced = funcol.all_reduce(x, "sum", group_name)
    h = reduced + residual
    normed = F.rms_norm(h, weight.shape, weight, eps)
    return normed, h  # h becomes the new residual
```

### Step 3: One-time setup during model initialization

```python
import torch.distributed._symmetric_memory as symm_mem

group_name = tp_group.group_name  # the tensor-parallel process group

# Pre-allocate the P2P workspace sized for the largest activation tensor
# that will go through the fused op. For decode: num_tokens × hidden_dim × 2 bytes.
max_tokens = max_batch_size  # or whatever the max decode batch is
workspace_bytes = max_tokens * hidden_dim * 2  # bf16 = 2 bytes
symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_bytes)
```

### Step 4: Call under inference mode

The FX pass only fires under `torch.inference_mode()`. SGLang's decode path
should already run under inference mode, but verify this.

```python
@torch.inference_mode()
def decode_forward(self, ...):
    ...
    normed, residual = fused_ar_norm(
        hidden_states, residual, self.norm_weight, group_name, eps=self.eps
    )
    ...
```

### Step 5: CUDA graph compatibility

The fused op is CUDA-graph safe. The first few eager warmup calls establish
the symmetric memory workspace and compile the Triton kernel. After that,
CUDA graph capture and replay work normally.

```python
# Standard SGLang CUDA graph capture pattern works unchanged:
# 1. Eager warmup (triggers Triton compilation + symm mem setup)
# 2. Capture graph
# 3. Replay
```

## Integration Path: Direct Op Call (Advanced)

For frameworks that don't use `torch.compile`, the fused op can be called
directly:

```python
import torch.distributed._symmetric_memory._fused_all_reduce_rmsnorm  # register ops

normed, new_residual = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
    hidden_states,      # input tensor (output of down_proj)
    norm_weight,        # RMSNorm weight parameter
    "sum",              # reduce op
    group_name,         # TP group name
    residual=residual,  # optional residual to add
    eps=1e-5,           # RMSNorm epsilon
)
```

This bypasses `torch.compile` entirely. The op handles workspace allocation,
P2P copy, barriers, and kernel launch internally.

## Memory Pool Zero-Copy (Optional Optimization)

For maximum performance, allocate the matmul output directly in symmetric
memory to eliminate the DtoD workspace copy:

```python
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._symmetric_memory._fused_allreduce_rmsnorm_triton import (
    _launch_fused_kernel,
    _make_peer_bufs,
)

# One-time setup (before CUDA graph capture):
mempool = symm_mem.get_mem_pool(device)
with torch.cuda.use_mem_pool(mempool):
    h_symm = torch.empty(max_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
sm_hdl = symm_mem.rendezvous(h_symm, tp_group)
peer_bufs = _make_peer_bufs(sm_hdl, h_symm.shape, h_symm.dtype)

# Per-iteration (CUDA-graph capturable):
torch.mm(intermediate, down_proj_weight.t(), out=h_symm)
output, residual_out = _launch_fused_kernel(
    sm_hdl, peer_bufs, h_symm, norm_weight, residual=residual, eps=eps,
)
```

This saves ~5-6us per call (the DtoD copy). The tradeoff is more integration
complexity: the upstream matmul must write directly into the pre-allocated
symmetric buffer via `out=`.

## API Reference

### `symm_mem.get_symm_mem_workspace(group_name, min_size=N)`

Pre-allocates a symmetric memory workspace of at least `N` bytes. Cached —
subsequent calls with the same or smaller `min_size` return the existing
workspace. **Must be called before the first fused op invocation.**

### `torch.ops.symm_mem.fused_all_reduce_rmsnorm(input, weight, reduce_op, group_name, *, residual=None, eps=1e-6)`

Returns `(normed, pre_norm)`:
- `normed`: RMS-normalized output, same shape/dtype as input
- `pre_norm`: `reduced + residual` if residual provided, else `None`

### Compile option: `"_fused_all_reduce_rmsnorm": True`

Enables the FX pattern matching pass. Off by default. Pass via
`torch.compile(options={...})`.

## What to Watch For

1. **`funcol.all_reduce` is required.** The FX pass matches
   `c10d_functional.all_reduce` → `c10d_functional.wait_tensor` nodes.
   Custom allreduce wrappers, `dist.all_reduce`, or NCCL calls won't match.

2. **`F.rms_norm` is required.** The FX pass matches the ATen decomposition
   of `F.rms_norm` (pow → mean → add → rsqrt → mul → mul). Custom RMSNorm
   implementations that don't go through `F.rms_norm` won't match.

3. **Inference mode only.** The FX pass is gated on inference mode. In
   training, AOTAutograd saves intermediates that the fused op can't produce.

4. **Single-node only.** The P2P loads require NVLink. Multi-node TP would
   need a different communication backend.

5. **Weight dtype.** The RMSNorm weight is typically fp32 while activations
   are bf16. The Triton kernel handles mixed precision internally (accumulates
   in fp32). If you see a warning about dtype mismatch from `torch.rms_norm`,
   it's harmless — the fused kernel handles it correctly.

6. **Workspace sizing.** The workspace must be large enough for the largest
   activation that will go through the fused op. For decode with variable
   batch sizes, size it for the maximum. The workspace is reused across calls.

## Testing

After integration, verify correctness by comparing outputs:

```python
# Reference: standard NCCL path
ref_reduced = funcol.all_reduce(x, "sum", group_name)
ref_h = ref_reduced + residual
ref_normed = F.rms_norm(ref_h, weight.shape, weight, eps)

# Fused path
fused_normed, fused_h = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
    x, weight, "sum", group_name, residual=residual, eps=eps,
)

torch.testing.assert_close(fused_normed, ref_normed, atol=1e-2, rtol=1e-2)
torch.testing.assert_close(fused_h, ref_h, atol=1e-2, rtol=1e-2)
```

Tolerances are slightly relaxed because the fused kernel accumulates in fp32
from bf16 inputs in a different order than the unfused path.

## Performance Expectations (Decode, 4× GPU)

For small decode tensors (1 token × 2880 hidden), the kernel itself is very
fast (<5us). The dominant costs are the two barrier synchronizations (~5-20us
each). The main win vs NCCL baseline is:

- **No NCCL overhead**: NCCL allreduce has significant latency for small
  messages (ring/tree protocol overhead). The P2P kernel does direct loads.
- **Fused memory traffic**: Intermediate tensors (reduced, reduced+residual)
  stay in registers instead of going through global memory.
- **Fewer kernel launches**: 3 kernels (barrier + fused + barrier) instead of
  5+ (NCCL + add + pow + mean + rsqrt + mul + mul).
