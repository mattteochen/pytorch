"""
Profile fused allreduce + RMSNorm: end-to-end SGLang-style repro.

Simulates a decoder layer: linear (MoE stand-in) → allreduce → residual add → RMSNorm.

Profiles five variants:
  1. baseline  – NCCL all_reduce + eager F.rms_norm
  2. fused_op  – torch.ops.symm_mem.fused_all_reduce_rmsnorm (direct, workspace copy)
  3. compiled  – torch.compile with FX pass fusion
  4. mempool   – mem pool zero-copy: matmul output lands in symmetric memory
  5. kraken    – single kernel with device-side sync (no barrier kernel launches)

Launch:
    torchrun --nproc_per_node=NUM_GPUS dev/profile_fused_allreduce_rmsnorm.py

Trace output: /tmp/fused_ar_rmsnorm_*.json
"""

import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed._symmetric_memory as symm_mem
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._symmetric_memory._fused_allreduce_rmsnorm_triton import (
    _launch_fused_kernel,
    _make_peer_bufs,
)

# Add kraken (third_party submodule) to import path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "kraken")
)
from kraken.fused.one_shot_all_reduce_bias_rms_norm import (
    one_shot_all_reduce_bias_rms_norm,
)

HIDDEN = 2880
NUM_TOKENS = 1  # decode tokens (flattened, no batch dim in SGLang)
INTER = 2880 * 4  # MoE/FFN intermediate dim
EPS = 1e-5
WARMUP_ITERS = 10
PROFILE_ITERS = 20


class DummyDecoderLayer(nn.Module):
    """Minimal decoder layer: linear (MoE stand-in) → allreduce → add residual → RMSNorm."""

    def __init__(self, hidden: int, inter: int, eps: float, device: torch.device):
        super().__init__()
        self.moe_proj = nn.Linear(hidden, inter, bias=False, device=device, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(inter, hidden, bias=False, device=device, dtype=torch.bfloat16)
        self.norm_weight = nn.Parameter(torch.ones(hidden, device=device, dtype=torch.float32))
        self.eps = eps

    def forward_baseline(self, x: torch.Tensor, residual: torch.Tensor, group_name: str):
        """NCCL all_reduce + eager RMSNorm (unfused baseline)."""
        h = self.down_proj(F.silu(self.moe_proj(x)))
        h = funcol.all_reduce(h, "sum", group_name)
        h = h + residual
        residual = h
        normed = F.rms_norm(h, self.norm_weight.shape, self.norm_weight, self.eps)
        return normed, residual


def _make_compiled_ar_norm(eps: float):
    """Build a torch.compile'd wrapper that the FX pass can fuse."""

    @torch.compile(
        options={"_fused_all_reduce_rmsnorm": True},
    )
    def _ar_norm(x, residual, weight, group_name):
        reduced = funcol.all_reduce(x, "sum", group_name)
        h = reduced + residual
        normed = F.rms_norm(h, weight.shape, weight, eps)
        return normed, h

    return _ar_norm


def _profile(name, fn, rank, warmup=WARMUP_ITERS, iters=PROFILE_ITERS, cuda_graph=False):
    # Eager warmup: Triton compilation, NCCL init, buffer allocation, etc.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if cuda_graph:
        # Side-stream warmup + capture (standard PyTorch pattern)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        stream.synchronize()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            fn()
        torch.cuda.synchronize()

        # Replay warmup so profiling sees steady-state only
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        run = g.replay
    else:
        run = fn

    trace_path = f"/tmp/fused_ar_rmsnorm_{name}_rank{rank}.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(iters):
            run()
            torch.cuda.synchronize()

    if rank == 0:
        prof.export_chrome_trace(trace_path)
        print(f"\n{'='*60}")
        print(f"  {name}  (trace: {trace_path})")
        print(f"{'='*60}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


@torch.inference_mode()
def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    group_name = dist.group.WORLD.group_name

    # --- Symmetric memory setup (mirrors what SGLang model init would do) ---
    symm_mem.enable_symm_mem_for_group(group_name)
    workspace_bytes = NUM_TOKENS * HIDDEN * 2  # bf16 = 2 bytes
    symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_bytes)
    if rank == 0:
        print(f"Symmetric memory workspace pre-allocated: {workspace_bytes} bytes")

    # --- Build dummy model ---
    layer = DummyDecoderLayer(HIDDEN, INTER, EPS, device)

    # --- Inputs (flattened, SGLang-style: num_tokens × hidden) ---
    x = torch.randn(NUM_TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = layer.norm_weight

    # --- Enable FX pass debug logging ---
    logging.getLogger("torch._inductor.fx_passes.fused_allreduce_rmsnorm").setLevel(
        logging.DEBUG
    )

    # --- Variant 1: baseline (NCCL + eager) ---
    _profile(
        "baseline",
        lambda: layer.forward_baseline(x, residual, group_name),
        rank,
        cuda_graph=True,
    )

    # --- Variant 2: fused op (direct call, no torch.compile) ---
    __import__("torch.distributed._symmetric_memory._fused_all_reduce_rmsnorm")

    def fused_op_call():
        h = layer.down_proj(F.silu(layer.moe_proj(x)))
        return torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            h, weight, "sum", group_name, residual=residual, eps=EPS,
        )

    _profile("fused_op", fused_op_call, rank, cuda_graph=True)

    # --- Variant 3: torch.compile with FX pass fusion ---
    ar_norm_compiled = _make_compiled_ar_norm(EPS)

    def compiled_call():
        h = layer.down_proj(F.silu(layer.moe_proj(x)))
        return ar_norm_compiled(h, residual, weight, group_name)

    _profile("compiled", compiled_call, rank, warmup=WARMUP_ITERS + 5, cuda_graph=True)

    # --- Variant 4: mem pool (zero-copy, no DtoD workspace copy) ---
    # Pre-allocate output in symmetric memory and rendezvous BEFORE capture
    # so the CUDA graph only sees GPU ops with fixed addresses.
    mempool = symm_mem.get_mem_pool(device)
    M_total = NUM_TOKENS
    with torch.cuda.use_mem_pool(mempool):
        h_symm = torch.empty(M_total, HIDDEN, device=device, dtype=torch.bfloat16)
    sm_hdl = symm_mem.rendezvous(h_symm, dist.group.WORLD)
    peer_bufs = _make_peer_bufs(sm_hdl, tuple(h_symm.shape), h_symm.dtype)

    def mempool_call():
        intermediate = F.silu(layer.moe_proj(x))
        torch.mm(intermediate.view(-1, INTER), layer.down_proj.weight.t(), out=h_symm)
        output, residual_out = _launch_fused_kernel(
            sm_hdl, peer_bufs, h_symm, weight, residual=residual, eps=EPS,
        )
        return output.view(x.shape), residual_out.view(x.shape)

    _profile("mempool", mempool_call, rank, cuda_graph=True)

    # --- Variant 5: kraken (device-side sync, single kernel launch) ---
    # Kraken needs a pre-allocated symmetric memory buffer + pre-allocated output.
    kraken_symm_buf = symm_mem.empty(
        (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device,
    )
    symm_mem.rendezvous(kraken_symm_buf, group=dist.group.WORLD)
    kraken_output = torch.empty_like(x)

    def kraken_call():
        h = layer.down_proj(F.silu(layer.moe_proj(x)))
        one_shot_all_reduce_bias_rms_norm(
            kraken_symm_buf, h, residual, weight, kraken_output, eps=EPS,
        )
        return kraken_output

    _profile("kraken", kraken_call, rank, cuda_graph=True)

    dist.destroy_process_group()
    if rank == 0:
        print("\nDone. Compare traces in /tmp/fused_ar_rmsnorm_*.json")


if __name__ == "__main__":
    main()
