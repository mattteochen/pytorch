"""
Profile fused allreduce + RMSNorm: end-to-end SGLang-style repro.

Simulates a decoder layer: linear (MoE stand-in) → allreduce → residual add → RMSNorm.

Profiles variants:
  1. baseline            – NCCL all_reduce + eager F.rms_norm
  2. fused_op            – torch.ops.symm_mem.fused_all_reduce_rmsnorm (direct, host-side barriers)
  3. compiled            – torch.compile with inductor P2P codegen (kraken device-side sync)
  3b. compiled_host_barrier – compiled + forced host barriers
  3c. compiled_gridcap36 – compiled + device CAS + grid cap at 36 CTAs
  3d. compiled_lamport   – compiled + Lamport push-model (zero barriers, inductor codegen)
  3e. compiled_2shot     – compiled + two-shot reduce-scatter+allgather (device CAS sync)
  4. compiled_plain      – torch.compile default settings (no fused allreduce+rmsnorm option)
  5. compiled_mempool    – compiled + mempool zero-copy (matmul → symm mem, single kernel)
  6. mempool             – mem pool zero-copy with handwritten kernel (host-side barriers)
  7. kraken              – kraken handwritten single kernel (one-shot, device-side sync)
  8. kraken_2shot        – kraken handwritten single kernel (two-shot, device-side sync)
  9. flashinfer          – FlashInfer trtllm_allreduce_fusion one-shot (no quant)
 10. lamport_standalone  – Lamport push-model standalone Triton kernel (reference)

Launch (torch profiler):
    torchrun --nproc_per_node=NUM_GPUS dev/profile_fused_allreduce_rmsnorm.py

Launch (wall-clock timer only, no profiler overhead):
    torchrun --nproc_per_node=NUM_GPUS dev/profile_fused_allreduce_rmsnorm.py --timer

Launch (nsys, NVTX markers only, no torch profiler):
    nsys profile torchrun --nproc_per_node=NUM_GPUS dev/profile_fused_allreduce_rmsnorm.py --nsys

Trace output: /tmp/fused_ar_rmsnorm_*.json
"""

import argparse
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


# Add dev/ and kraken to import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "kraken"
    ),
)
from kraken.fused.one_shot_all_reduce_bias_rms_norm import (
    one_shot_all_reduce_bias_rms_norm,
)
from kraken.fused.two_shot_all_reduce_bias_rms_norm import (
    two_shot_all_reduce_bias_rms_norm,
)
from compile import compile_with_debug


HIDDEN = 2880
NUM_TOKENS = 1  # decode tokens (flattened, no batch dim in SGLang)
INTER = 2880 * 4  # MoE/FFN intermediate dim
EPS = 1e-5
WARMUP_ITERS = 10
PROFILE_ITERS = 100

torch.cuda.cudart().cudaProfilerStart()


class DummyDecoderLayer(nn.Module):
    """Minimal decoder layer: linear (MoE stand-in) → allreduce → add residual → RMSNorm."""

    def __init__(self, hidden: int, inter: int, eps: float, device: torch.device):
        super().__init__()
        self.moe_proj = nn.Linear(
            hidden, inter, bias=False, device=device, dtype=torch.bfloat16
        )
        self.down_proj = nn.Linear(
            inter, hidden, bias=False, device=device, dtype=torch.bfloat16
        )
        self.norm_weight = nn.Parameter(
            torch.ones(hidden, device=device, dtype=torch.float32)
        )
        self.eps = eps

    def forward_baseline(
        self, x: torch.Tensor, residual: torch.Tensor, group_name: str
    ):
        """NCCL all_reduce + eager RMSNorm (unfused baseline)."""
        # h = self.down_proj(F.silu(self.moe_proj(x)))
        h = x
        h = funcol.all_reduce(h, "sum", group_name)
        h = h + residual
        residual = h
        normed = F.rms_norm(h, self.norm_weight.shape, self.norm_weight, self.eps)
        return normed, residual


def _make_compiled_ar_norm(eps: float):
    """Inductor P2P kernel with device-side CAS sync (original, no host barriers)."""

    @torch.compile(options={
        "_fuse_symm_mem_comms": True,
        "_symm_mem_host_barrier_threshold": -1,
    })
    def _ar_norm(x, residual, weight, group_name):
        reduced = funcol.all_reduce(x, "sum", group_name)
        h = reduced + residual
        normed = F.rms_norm(h, weight.shape, weight, eps)
        return normed, h

    return _ar_norm


def _make_compiled_plain_ar_norm(eps: float):
    """Inductor default compile path without fused allreduce+rmsnorm option."""

    @torch.compile
    def _ar_norm_plain(x, residual, weight, group_name):
        reduced = funcol.all_reduce(x, "sum", group_name)
        h = reduced + residual
        normed = F.rms_norm(h, weight.shape, weight, eps)
        return normed, h

    return _ar_norm_plain


def _profile(
    name,
    fn,
    rank,
    warmup=WARMUP_ITERS,
    iters=PROFILE_ITERS,
    cuda_graph=False,
    nsys_mode=False,
    timer_mode=False,
) -> float | None:
    """Returns avg_us when timer_mode is True, else None."""
    import time

    CUDA_GRAPH_ITERS = 10

    # Eager warmup: Triton compilation, NCCL init, buffer allocation, etc.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if cuda_graph:
        # Side-stream capture (standard PyTorch pattern)
        stream = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(stream)

        with torch.cuda.graph(g, stream=stream):
            for _ in range(CUDA_GRAPH_ITERS):
                fn()
        torch.cuda.synchronize()

        # Replay warmup so profiling sees steady-state only
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        run = g.replay
    else:
        run = fn

    # Always measure wall-clock time.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if nsys_mode:
        torch.cuda.nvtx.range_push(name)
    for _ in range(iters // CUDA_GRAPH_ITERS if cuda_graph else iters):
        run()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg_us = (t1 - t0) / iters * 1e6
    if nsys_mode:
        torch.cuda.nvtx.range_pop()

    if not timer_mode and not nsys_mode:
        # Torch profiler pass (re-run to capture trace).
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
            print(f"\n{'=' * 60}")
            print(f"  {name}  (trace: {trace_path})")
            print(f"{'=' * 60}")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    return avg_us


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="NVTX-only mode: skip torch profiler, emit NVTX ranges for nsys",
    )
    parser.add_argument(
        "--timer",
        action="store_true",
        help="Wall-clock timer only: no profiler, no NVTX, just elapsed time",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[NUM_TOKENS],
        help="Number of decode tokens (one or more values, e.g. --num-tokens 1 32 1024)",
    )
    # torchrun injects extra args; ignore them.
    args, _ = parser.parse_known_args()
    nsys_mode = args.nsys
    timer_mode = args.timer
    token_counts = args.num_tokens

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    group_name = dist.group.WORLD.group_name

    # --- Symmetric memory setup (mirrors what SGLang model init would do) ---
    symm_mem.enable_symm_mem_for_group(group_name)
    max_tokens = max(token_counts)
    workspace_bytes = max_tokens * HIDDEN * 2  # bf16 = 2 bytes
    symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_bytes)
    # all_results: {num_tokens: [(name, avg_us, correctness), ...]}
    all_results: dict[int, list[tuple[str, float | None, str]]] = {}

    if rank == 0:
        mode_str = (
            "timer"
            if timer_mode
            else ("nsys (NVTX only)" if nsys_mode else "torch profiler")
        )
        print(f"Mode: {mode_str}")
        print(
            f"Config: num_tokens={token_counts}, HIDDEN={HIDDEN}, world_size={dist.get_world_size()}"
        )
        print(f"Symmetric memory workspace pre-allocated: {workspace_bytes} bytes")

    # --- Build dummy model ---
    layer = DummyDecoderLayer(HIDDEN, INTER, EPS, device)

    # --- Enable FX pass debug logging ---
    logging.getLogger("torch._inductor.fx_passes.fused_allreduce_rmsnorm").setLevel(
        logging.DEBUG
    )

    # --- Correctness checking infrastructure ---
    def _extract_normed(result):
        """Extract the normed output tensor from a variant's return value."""
        if isinstance(result, tuple):
            return result[0]
        return result

    def _check_correctness(fn, ref_normed, label):
        """Run fn once and compare normed output against reference.

        Returns a status string: "PASS", "FAIL (atol=X)", or "ERROR: ...".
        """
        try:
            out = fn()
            normed = _extract_normed(out)
            if normed is None:
                return "SKIP"
            if torch.allclose(normed, ref_normed, atol=1e-2, rtol=1e-2):
                return "PASS"
            diff = (normed.float() - ref_normed.float()).abs()
            return f"FAIL (max={diff.max().item():.4f}, mean={diff.mean().item():.4f})"
        except Exception as e:
            return f"ERROR: {e}"

    def _print_summary(results, num_tokens):
        """Print per-token-count summary table."""
        baseline_us = next(
            (us for name, us, _ in results if name == "baseline" and us), None
        )
        print(f"\n{'=' * 80}")
        print(
            f"  SUMMARY  (NUM_TOKENS={num_tokens}, HIDDEN={HIDDEN}, "
            f"world_size={dist.get_world_size()})"
        )
        print(f"{'=' * 80}")
        print(f"  {'Variant':<25s} {'us/iter':>10s} {'vs baseline':>12s}  {'correctness'}")
        print(f"  {'-' * 25} {'-' * 10} {'-' * 12}  {'-' * 28}")
        for name, avg_us, ok in results:
            if avg_us is None:
                continue
            speedup = ""
            if baseline_us and baseline_us > 0:
                ratio = baseline_us / avg_us
                speedup = f"{ratio:.2f}x"
            print(f"  {name:<25s} {avg_us:10.1f} {speedup:>12s}  {ok}")

    def _run_benchmarks_for_token_count(num_tokens, x, residual, weight):
        """Run all benchmark variants for a given token count, return results list."""
        results: list[tuple[str, float | None, str]] = []

        # Run once to get reference output for correctness checks.
        ref_out = layer.forward_baseline(x, residual, group_name)
        torch.cuda.synchronize()
        ref_normed = _extract_normed(ref_out).clone()

        results.append(
            (
                "baseline",
                _profile(
                    "baseline",
                    lambda: layer.forward_baseline(x, residual, group_name),
                    rank,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                "REF",
            )
        )

        # --- Variant 2: fused op (direct call, no torch.compile) ---
        __import__("torch.distributed._symmetric_memory._fused_all_reduce_rmsnorm")

        def fused_op_call():
            # h = layer.down_proj(F.silu(layer.moe_proj(x)))
            h = x
            return torch.ops.symm_mem.fused_all_reduce_rmsnorm(
                h,
                weight,
                "sum",
                group_name,
                residual=residual,
                eps=EPS,
            )

        fused_op_ok = _check_correctness(fused_op_call, ref_normed, "fused_op")
        results.append(
            (
                "fused_op",
                _profile(
                    "fused_op",
                    fused_op_call,
                    rank,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                fused_op_ok,
            )
        )

        # --- Variant 3: torch.compile (inductor P2P codegen, kraken sync) ---
        ar_norm_compiled = _make_compiled_ar_norm(EPS)

        def compiled_call():
            # h = layer.down_proj(F.silu(layer.moe_proj(x)))
            h = x
            return ar_norm_compiled(h, residual, weight, group_name)

        compiled_ok = _check_correctness(compiled_call, ref_normed, "compiled")
        results.append(
            (
                "compiled",
                _profile(
                    "compiled",
                    compiled_call,
                    rank,
                    warmup=WARMUP_ITERS + 5,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                compiled_ok,
            )
        )

        # --- Variant 3b: compiled + forced host barriers ---
        def _make_host_barrier_ar_norm(eps_val):
            @torch.compile(options={
                "_fuse_symm_mem_comms": True,
                "_symm_mem_host_barrier_threshold": 0,
            })
            def _fn(x, residual, weight, group_name):
                reduced = funcol.all_reduce(x, "sum", group_name)
                h = reduced + residual
                normed = F.rms_norm(h, weight.shape, weight, eps_val)
                return normed, h
            return _fn

        ar_norm_host_barrier = _make_host_barrier_ar_norm(EPS)

        def compiled_host_barrier_call():
            h = x
            return ar_norm_host_barrier(h, residual, weight, group_name)

        host_barrier_ok = _check_correctness(compiled_host_barrier_call, ref_normed, "compiled_host_barrier")
        results.append(
            (
                "compiled_host_barrier",
                _profile(
                    "compiled_host_barrier",
                    compiled_host_barrier_call,
                    rank,
                    warmup=WARMUP_ITERS + 5,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                host_barrier_ok,
            )
        )

        # --- Variant 3c: compiled + device CAS + grid cap at 36 CTAs ---
        def _make_grid_cap_ar_norm(eps_val):
            @torch.compile(options={
                "_fuse_symm_mem_comms": True,
                "_symm_mem_host_barrier_threshold": -1,
                "_symm_mem_grid_cap": 36,
            })
            def _fn(x, residual, weight, group_name):
                reduced = funcol.all_reduce(x, "sum", group_name)
                h = reduced + residual
                normed = F.rms_norm(h, weight.shape, weight, eps_val)
                return normed, h
            return _fn

        ar_norm_grid_cap = _make_grid_cap_ar_norm(EPS)

        def compiled_grid_cap_call():
            h = x
            return ar_norm_grid_cap(h, residual, weight, group_name)

        gridcap_ok = _check_correctness(compiled_grid_cap_call, ref_normed, "compiled_gridcap36")
        results.append(
            (
                "compiled_gridcap36",
                _profile(
                    "compiled_gridcap36",
                    compiled_grid_cap_call,
                    rank,
                    warmup=WARMUP_ITERS + 5,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                gridcap_ok,
            )
        )

        # --- Variant 3d: compiled + Lamport push-model (zero barriers) ---
        def _make_lamport_ar_norm(eps_val):
            def _fn(x, residual, weight, group_name):
                reduced = funcol.all_reduce(x, "sum", group_name)
                h = reduced + residual
                normed = F.rms_norm(h, weight.shape, weight, eps_val)
                return normed, h
            return torch.compile(_fn, options={
                "_fuse_symm_mem_comms": True,
                "_symm_mem_sync_mode": "lamport",
            })

        ar_norm_lamport = _make_lamport_ar_norm(EPS)

        def compiled_lamport_call():
            h = x
            return ar_norm_lamport(h, residual, weight, group_name)

        lamport_ok = _check_correctness(compiled_lamport_call, ref_normed, "compiled_lamport")
        results.append(
            (
                "compiled_lamport",
                _profile(
                    "compiled_lamport",
                    compiled_lamport_call,
                    rank,
                    warmup=WARMUP_ITERS + 5,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                lamport_ok,
            )
        )

        # --- Variant 3e: compiled + two-shot reduce-scatter+allgather ---
        def _make_2shot_ar_norm(eps_val):
            def _fn(x, residual, weight, group_name):
                reduced = funcol.all_reduce(x, "sum", group_name)
                h = reduced + residual
                normed = F.rms_norm(h, weight.shape, weight, eps_val)
                return normed, h
            return torch.compile(_fn, options={
                "_fuse_symm_mem_comms": True,
                "_fuse_symm_mem_comms_max_bytes": 0,
                "_symm_mem_sync_mode": "device_cas_2_shot",
                "_symm_mem_grid_cap": 128,
                "trace.enabled": True,
            })

        ar_norm_2shot = _make_2shot_ar_norm(EPS)

        def compiled_2shot_call():
            h = x
            return ar_norm_2shot(h, residual, weight, group_name)

        two_shot_ok = _check_correctness(compiled_2shot_call, ref_normed, "compiled_2shot")
        results.append(
            (
                "compiled_2shot",
                _profile(
                    "compiled_2shot",
                    compiled_2shot_call,
                    rank,
                    warmup=WARMUP_ITERS + 5,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                two_shot_ok,
            )
        )

        # --- Variant 4: torch.compile plain (default options) ---
        ar_norm_compiled_plain = _make_compiled_plain_ar_norm(EPS)

        def compiled_plain_call():
            h = x
            return ar_norm_compiled_plain(h, residual, weight, group_name)

        plain_ok = _check_correctness(compiled_plain_call, ref_normed, "compiled_plain")
        results.append(
            (
                "compiled_plain",
                _profile(
                    "compiled_plain",
                    compiled_plain_call,
                    rank,
                    warmup=WARMUP_ITERS + 5,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                plain_ok,
            )
        )

        # --- Variant 5: compiled + mempool (zero-copy matmul -> symm mem) ---
        # The matmul output lands directly in symmetric memory via the mem pool.
        # _symm_mem_skip_prologue_copy tells the codegen to skip the copy
        # (input is already where peers can read it) and just sync.

        mempool_for_compiled = symm_mem.get_mem_pool(device)
        with torch.cuda.use_mem_pool(mempool_for_compiled):
            h_symm_compiled = torch.empty(
                num_tokens, HIDDEN, device=device, dtype=torch.bfloat16
            )
        symm_mem.rendezvous(h_symm_compiled, dist.group.WORLD)

        ar_norm_compiled_mp = _make_compiled_ar_norm(EPS)

        # Compile with skip_prologue_copy so the kernel omits the copy.
        # We need to trigger recompilation with the new config, so we
        # create a fresh compiled function.
        @torch.compile(
            options={
                "_fuse_symm_mem_comms": True,
                "_symm_mem_skip_prologue_copy": True,
            },
        )
        def _ar_norm_mp(x, residual, w, group_name):
            reduced = funcol.all_reduce(x, "sum", group_name)
            h = reduced + residual
            normed = F.rms_norm(h, w.shape, w, EPS)
            return normed, h

        def compiled_mempool_call():
            # intermediate = F.silu(layer.moe_proj(x))
            # torch.mm(
            #     intermediate.view(-1, INTER),
            #     layer.down_proj.weight.t(),
            #     out=h_symm_compiled,
            # )
            h_symm_compiled.copy_(x)
            return _ar_norm_mp(h_symm_compiled, residual, weight, group_name)

        mempool_compiled_ok = _check_correctness(compiled_mempool_call, ref_normed, "compiled_mempool")
        results.append(
            (
                "compiled_mempool",
                _profile(
                    "compiled_mempool",
                    compiled_mempool_call,
                    rank,
                    warmup=WARMUP_ITERS + 5,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                mempool_compiled_ok,
            )
        )

        # --- Variant 6: mem pool (zero-copy with handwritten kernel) ---
        # (Uses the old handwritten Triton kernel with host-side barriers)
        # Pre-allocate output in symmetric memory and rendezvous BEFORE capture
        # so the CUDA graph only sees GPU ops with fixed addresses.
        mempool = symm_mem.get_mem_pool(device)
        M_total = num_tokens
        with torch.cuda.use_mem_pool(mempool):
            h_symm = torch.empty(M_total, HIDDEN, device=device, dtype=torch.bfloat16)
        sm_hdl = symm_mem.rendezvous(h_symm, dist.group.WORLD)
        peer_bufs = _make_peer_bufs(sm_hdl, tuple(h_symm.shape), h_symm.dtype)

        def mempool_call():
            # intermediate = F.silu(layer.moe_proj(x))
            # torch.mm(intermediate.view(-1, INTER), layer.down_proj.weight.t(), out=h_symm)
            h_symm.copy_(x)
            output, residual_out = _launch_fused_kernel(
                sm_hdl,
                peer_bufs,
                h_symm,
                weight,
                residual=residual,
                eps=EPS,
            )
            return output.view(x.shape), residual_out.view(x.shape)

        mempool_ok = _check_correctness(mempool_call, ref_normed, "mempool")
        results.append(
            (
                "mempool",
                _profile(
                    "mempool",
                    mempool_call,
                    rank,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                mempool_ok,
            )
        )

        # --- Variant 7: kraken (device-side sync, single kernel launch) ---
        # Kraken needs a pre-allocated symmetric memory buffer + pre-allocated output.
        kraken_symm_buf = symm_mem.empty(
            (num_tokens, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        symm_mem.rendezvous(kraken_symm_buf, group=dist.group.WORLD)
        kraken_output = torch.empty_like(x)

        def kraken_call():
            # h = layer.down_proj(F.silu(layer.moe_proj(x)))
            h = x
            one_shot_all_reduce_bias_rms_norm(
                kraken_symm_buf,
                h,
                residual,
                weight,
                kraken_output,
                eps=EPS,
            )
            return kraken_output

        kraken_ok = _check_correctness(kraken_call, ref_normed, "kraken")
        results.append(
            (
                "kraken",
                _profile(
                    "kraken",
                    kraken_call,
                    rank,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                kraken_ok,
            )
        )

        # --- Variant 8: kraken two-shot (device-side sync, single kernel launch) ---
        kraken_2shot_symm_buf = symm_mem.empty(
            (num_tokens, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        symm_mem.rendezvous(kraken_2shot_symm_buf, group=dist.group.WORLD)
        kraken_2shot_output = torch.empty_like(x)

        def kraken_2shot_call():
            h = x
            two_shot_all_reduce_bias_rms_norm(
                kraken_2shot_symm_buf,
                h,
                residual,
                weight,
                kraken_2shot_output,
                eps=EPS,
            )
            return kraken_2shot_output

        kraken_2shot_ok = _check_correctness(kraken_2shot_call, ref_normed, "kraken_2shot")
        results.append(
            (
                "kraken_2shot",
                _profile(
                    "kraken_2shot",
                    kraken_2shot_call,
                    rank,
                    cuda_graph=True,
                    nsys_mode=nsys_mode,
                    timer_mode=timer_mode,
                ),
                kraken_2shot_ok,
            )
        )

        # --- Variant 9: FlashInfer trtllm_allreduce_fusion (one-shot, no quant) ---
        try:
            import flashinfer.comm as flashinfer_comm

            fi_max_token = 4096
            fi_ipc_handles, fi_workspace = (
                flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                    tp_rank=rank,
                    tp_size=dist.get_world_size(),
                    max_token_num=fi_max_token,
                    hidden_dim=HIDDEN,
                    group=dist.group.WORLD,
                )
            )
            fi_norm_out = torch.empty_like(x)
            fi_residual_out = torch.empty_like(x)

            def flashinfer_call():
                # h = layer.down_proj(F.silu(layer.moe_proj(x)))
                h = x
                flashinfer_comm.trtllm_allreduce_fusion(
                    allreduce_in=h,
                    token_num=h.shape[0],
                    residual_in=residual,
                    residual_out=fi_residual_out,
                    norm_out=fi_norm_out,
                    rms_gamma=weight,
                    rms_eps=EPS,
                    hidden_dim=HIDDEN,
                    workspace_ptrs=fi_workspace,
                    pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                    allreduce_out=None,
                    quant_out=None,
                    scale_out=None,
                    layout_code=None,
                    scale_factor=None,
                    use_oneshot=None,
                    world_rank=rank,
                    world_size=dist.get_world_size(),
                    launch_with_pdl=True,
                    trigger_completion_at_end=True,
                    fp32_acc=True,
                )
                return fi_norm_out

            flashinfer_ok = _check_correctness(flashinfer_call, ref_normed, "flashinfer")
            results.append(
                (
                    "flashinfer",
                    _profile(
                        "flashinfer",
                        flashinfer_call,
                        rank,
                        cuda_graph=True,
                        nsys_mode=nsys_mode,
                        timer_mode=timer_mode,
                    ),
                    flashinfer_ok,
                )
            )

            flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(
                fi_ipc_handles, dist.group.WORLD
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            if rank == 0:
                print(f"\nSkipping FlashInfer variant: {e}")

        # --- Variant 10: Lamport push-model standalone (reference, not inductor) ---
        try:
            from lamport_allreduce_rmsnorm import (
                lamport_allreduce_rmsnorm,
                setup_lamport_workspace,
            )

            _lam_sm, _lam_buf, _lam_buf_ptrs, _lam_slot_elems = setup_lamport_workspace(
                device, num_tokens, HIDDEN, dist.get_world_size(), dist.group.WORLD,
            )
            _lam_iter = [0]

            def lamport_standalone_call():
                h = x
                r = lamport_allreduce_rmsnorm(
                    h, weight, _lam_buf_ptrs, _lam_slot_elems, _lam_iter[0],
                    rank, dist.get_world_size(), residual=residual, eps=EPS,
                )
                _lam_iter[0] += 1
                return r

            lamport_sa_ok = _check_correctness(lamport_standalone_call, ref_normed, "lamport_standalone")
            results.append(
                (
                    "lamport_standalone",
                    _profile(
                        "lamport_standalone",
                        lamport_standalone_call,
                        rank,
                        cuda_graph=True,
                        nsys_mode=nsys_mode,
                        timer_mode=timer_mode,
                    ),
                    lamport_sa_ok,
                )
            )
        except (ImportError, RuntimeError) as e:
            if rank == 0:
                print(f"\nSkipping Lamport standalone variant: {e}")

        dist.barrier()
        torch.cuda.synchronize()
        return results

    # --- Main loop: run benchmarks for each token count ---
    for num_tokens in token_counts:
        dist.barrier()
        torch.cuda.synchronize()

        if rank == 0:
            print(f"\n{'#' * 80}")
            print(f"  NUM_TOKENS = {num_tokens}")
            print(f"{'#' * 80}")

        torch._dynamo.reset()

        # Clear symm_mem / lamport caches so setup functions re-allocate
        # for the new tensor shape (these are collectives — all ranks must
        # call them together, so stale cache hits would desync ranks).
        from torch._inductor.runtime.symm_mem_helpers import _symm_mem_cache
        from torch._inductor.runtime.lamport_helpers import _lamport_cache
        _symm_mem_cache.clear()
        _lamport_cache.clear()

        x = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = layer.norm_weight

        results = _run_benchmarks_for_token_count(num_tokens, x, residual, weight)
        all_results[num_tokens] = results

        if rank == 0:
            _print_summary(results, num_tokens)

    # --- Combined table across all token counts ---
    if rank == 0 and len(token_counts) > 1:
        # Collect all variant names in order of first appearance
        variant_names = []
        for tc in token_counts:
            for name, _, _ in all_results.get(tc, []):
                if name not in variant_names:
                    variant_names.append(name)

        print(f"\n{'=' * 80}")
        print(
            f"  COMBINED  (HIDDEN={HIDDEN}, "
            f"world_size={dist.get_world_size()})"
        )
        print(f"{'=' * 80}")
        header = f"  {'Variant':<25s}"
        for tc in token_counts:
            header += f" {'T=' + str(tc):>12s}"
        print(header)
        print(f"  {'-' * 25}" + f" {'-' * 12}" * len(token_counts))
        for vname in variant_names:
            row = f"  {vname:<25s}"
            for tc in token_counts:
                us = None
                for name, avg_us, _ in all_results.get(tc, []):
                    if name == vname and avg_us is not None:
                        us = avg_us
                        break
                if us is not None:
                    row += f" {us:12.1f}"
                else:
                    row += f" {'--':>12s}"
            print(row)
        print()

    dist.destroy_process_group()
    if rank == 0:
        print("Done. Compare traces in /tmp/fused_ar_rmsnorm_*.json")

    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
