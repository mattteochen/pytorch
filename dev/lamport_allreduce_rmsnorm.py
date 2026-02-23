"""
Lamport push-model allreduce + RMSNorm: standalone Triton kernel.

Implements FlashInfer-style barrier-free P2P allreduce using the Lamport
sentinel protocol (-0.0 as sentinel, push model, volatile polling).
Fuses allreduce + residual add + RMSNorm into a single kernel launch
with zero barriers and zero atomics.

Launch:
    torchrun --nproc_per_node=4 dev/lamport_allreduce_rmsnorm.py
    torchrun --nproc_per_node=4 dev/lamport_allreduce_rmsnorm.py --num-tokens 1024
    torchrun --nproc_per_node=4 dev/lamport_allreduce_rmsnorm.py --timer
"""

import argparse
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEG_ZERO_U16 = 0x8000
_NEG_ZERO = tl.constexpr(0x8000)

# ---------------------------------------------------------------------------
# Triton JIT helpers
# ---------------------------------------------------------------------------


@triton.jit
def _fence_sys():
    """System-scope fence ensuring all prior stores are visible to all GPUs."""
    tl.inline_asm_elementwise(
        "fence.sc.sys;", "=r", [], dtype=tl.int32, is_pure=False, pack=1,
    )


@triton.jit
def _volatile_load_u32_scalar(addr):
    """Single scalar volatile load bypassing L1 cache."""
    return tl.inline_asm_elementwise(
        "ld.volatile.global.b32 $0, [$1];",
        "=r, l", [addr], dtype=tl.uint32, is_pure=False, pack=1,
    )


@triton.jit
def _remove_neg_zero(val):
    """Replace bf16 -0.0 with +0.0 so real data never matches sentinel."""
    bits = val.to(tl.uint16, bitcast=True)
    return tl.where(bits == _NEG_ZERO, tl.zeros_like(val), val)


@triton.jit
def _poll_last_word(slot_u32_ptr, n_words):
    """Spin on the last u32 word of a slot until it contains no sentinel.

    Since the writer stores sequentially with a system fence, if the
    last word is ready, all prior words are guaranteed visible.
    """
    last_addr = slot_u32_ptr + (n_words - 1)
    ready = tl.full([], 0, dtype=tl.int32)
    while ready == 0:
        w = _volatile_load_u32_scalar(last_addr)
        lo = w & 0xFFFF
        hi = (w >> 16) & 0xFFFF
        ready = ((lo != _NEG_ZERO) & (hi != _NEG_ZERO)).to(tl.int32)


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------


@triton.jit
def lamport_allreduce_rmsnorm_kernel(
    input_ptr, output_ptr, residual_ptr, residual_out_ptr, weight_ptr,
    buf_ptrs,
    M, N, eps,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BUF_OFFSET: tl.constexpr,
    CLEAR_OFFSET: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    chunk = M * N

    # --- Phase 1: Push local data to all peers ---
    data = tl.load(input_ptr + row * N + cols, mask=mask, other=0.0)
    data = _remove_neg_zero(data)

    buf_ptrs_u64 = buf_ptrs.to(tl.pointer_type(tl.uint64))
    for peer in tl.static_range(WORLD_SIZE):
        peer_buf = tl.load(buf_ptrs_u64 + peer).to(tl.pointer_type(tl.bfloat16))
        tl.store(peer_buf + BUF_OFFSET + RANK * chunk + row * N + cols,
                 data, mask=mask)
    _fence_sys()

    # --- Phase 2: Poll own local buffer for all peers' data ---
    my_buf = tl.load(buf_ptrs_u64 + RANK).to(tl.pointer_type(tl.bfloat16))
    my_buf_base = my_buf + BUF_OFFSET
    n_words = N // 2

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for peer in tl.static_range(WORLD_SIZE):
        slot_bf16 = my_buf_base + peer * chunk + row * N
        slot_u32 = slot_bf16.to(tl.pointer_type(tl.uint32))
        _poll_last_word(slot_u32, n_words)
        val = tl.load(slot_bf16 + cols, mask=mask, other=0.0)
        acc += val.to(tl.float32)

    # --- Phase 3: Add residual + RMSNorm ---
    if HAS_RESIDUAL:
        res = tl.load(residual_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        acc = acc + res
        tl.store(residual_out_ptr + row * N + cols, acc.to(tl.bfloat16), mask=mask)

    variance = tl.sum(acc * acc, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(variance + eps)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = (acc * inv_rms * w).to(tl.bfloat16)
    tl.store(output_ptr + row * N + cols, normed, mask=mask)

    # --- Phase 4: Clear the old buffer slot ---
    neg_zero = tl.full([BLOCK_N], _NEG_ZERO, dtype=tl.uint16).to(tl.bfloat16, bitcast=True)
    clear_base = my_buf + CLEAR_OFFSET
    for peer in tl.static_range(WORLD_SIZE):
        clear_bf16 = (clear_base + peer * chunk + row * N).to(tl.pointer_type(tl.bfloat16))
        tl.store(clear_bf16 + cols, neg_zero, mask=mask)


# ---------------------------------------------------------------------------
# Python launcher
# ---------------------------------------------------------------------------


def setup_lamport_workspace(device, max_M, N, world_size, group):
    """Allocate triple-buffered symmetric memory workspace."""
    slot_elems = world_size * max_M * N
    total_elems = 3 * slot_elems
    buf = symm_mem.empty(total_elems, dtype=torch.bfloat16, device=device)
    sm = symm_mem.rendezvous(buf, group)
    buf.view(torch.uint16).fill_(NEG_ZERO_U16)
    sm.barrier(channel=0)
    buf_ptrs = torch.tensor(
        [sm.buffer_ptrs[i] for i in range(world_size)],
        dtype=torch.int64, device=device,
    )
    return sm, buf, buf_ptrs, slot_elems


def lamport_allreduce_rmsnorm(
    input_tensor, weight, buf_ptrs, slot_elems, iteration, rank, world_size,
    *, residual=None, eps=1e-6,
):
    M, N = input_tensor.shape
    output = torch.empty_like(input_tensor)
    has_residual = residual is not None
    residual_out = torch.empty_like(input_tensor) if has_residual else output
    BLOCK_N = triton.next_power_of_2(N)
    buf_offset = (iteration % 3) * slot_elems
    clear_offset = ((iteration + 2) % 3) * slot_elems
    lamport_allreduce_rmsnorm_kernel[(M,)](
        input_tensor, output,
        residual if has_residual else output, residual_out, weight, buf_ptrs,
        M, N, eps,
        RANK=rank, WORLD_SIZE=world_size,
        BUF_OFFSET=buf_offset, CLEAR_OFFSET=clear_offset,
        HAS_RESIDUAL=has_residual, BLOCK_N=BLOCK_N,
    )
    return output, residual_out if has_residual else None


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def reference_allreduce_rmsnorm(x, weight, eps, residual=None):
    group_name = dist.group.WORLD.group_name
    reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
    reduced = torch.ops._c10d_functional.wait_tensor(reduced)
    pre_norm = None
    if residual is not None:
        reduced = reduced + residual
        pre_norm = reduced.clone()
    normed = F.rms_norm(reduced, weight.shape, weight, eps)
    return normed, pre_norm


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

HIDDEN = 2880
NUM_TOKENS = 1
EPS = 1e-5
WARMUP = 10
ITERS = 100


def _profile(name, fn, warmup=WARMUP, iters=ITERS, cuda_graph=True):
    GRAPH_BATCH = 10
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if cuda_graph:
        stream = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(g, stream=stream):
            for _ in range(GRAPH_BATCH):
                fn()
        torch.cuda.synchronize()
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        run = g.replay
    else:
        run = fn
        GRAPH_BATCH = 1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters // GRAPH_BATCH):
        run()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timer", action="store_true")
    parser.add_argument("--num-tokens", type=int, default=NUM_TOKENS)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    args, _ = parser.parse_known_args()
    num_tokens, hidden = args.num_tokens, args.hidden

    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.ones(hidden, device=device, dtype=torch.float32)

    sm, buf, buf_ptrs, slot_elems = setup_lamport_workspace(
        device, num_tokens, hidden, world_size, dist.group.WORLD,
    )

    # --- Correctness ---
    expected_normed, expected_pre_norm = reference_allreduce_rmsnorm(
        x, weight, EPS, residual=residual,
    )
    iteration = [0]

    def lamport_call():
        r = lamport_allreduce_rmsnorm(
            x, weight, buf_ptrs, slot_elems, iteration[0],
            rank, world_size, residual=residual, eps=EPS,
        )
        iteration[0] += 1
        return r

    normed, pre_norm = lamport_call()
    torch.cuda.synchronize()
    if rank == 0:
        n_ok = torch.allclose(normed, expected_normed, atol=2e-2, rtol=2e-2)
        p_ok = torch.allclose(pre_norm, expected_pre_norm, atol=2e-2, rtol=2e-2)
        print(f"Correctness: normed={'PASS' if n_ok else 'FAIL'}, "
              f"pre_norm={'PASS' if p_ok else 'FAIL'}")
        if not n_ok:
            print(f"  normed max diff: {(normed - expected_normed).abs().max().item()}")
        if not p_ok:
            print(f"  pre_norm max diff: {(pre_norm - expected_pre_norm).abs().max().item()}")

    # --- Benchmark ---
    buf.view(torch.uint16).fill_(NEG_ZERO_U16)
    sm.barrier(channel=0)
    iteration[0] = 0
    lamport_us = _profile("lamport", lamport_call, cuda_graph=True)

    import torch.distributed._functional_collectives as funcol

    def baseline_call():
        reduced = funcol.all_reduce(x, "sum", group_name)
        h = reduced + residual
        return F.rms_norm(h, weight.shape, weight, EPS), h

    baseline_us = _profile("baseline", baseline_call, cuda_graph=True)

    __import__("torch.distributed._symmetric_memory._fused_all_reduce_rmsnorm")
    symm_mem.get_symm_mem_workspace(group_name, min_size=x.numel() * x.element_size())

    def fused_op_call():
        return torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x, weight, "sum", group_name, residual=residual, eps=EPS,
        )

    fused_op_us = _profile("fused_op", fused_op_call, cuda_graph=True)

    fi_us = None
    try:
        import flashinfer.comm as flashinfer_comm
        fi_handles, fi_ws = flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            tp_rank=rank, tp_size=world_size, max_token_num=4096,
            hidden_dim=hidden, group=dist.group.WORLD,
        )
        fi_norm_out, fi_res_out = torch.empty_like(x), torch.empty_like(x)

        def fi_call():
            flashinfer_comm.trtllm_allreduce_fusion(
                allreduce_in=x, token_num=x.shape[0],
                residual_in=residual, residual_out=fi_res_out,
                norm_out=fi_norm_out, rms_gamma=weight, rms_eps=EPS,
                hidden_dim=hidden, workspace_ptrs=fi_ws,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                allreduce_out=None, quant_out=None, scale_out=None,
                layout_code=None, scale_factor=None, use_oneshot=None,
                world_rank=rank, world_size=world_size,
                launch_with_pdl=True, trigger_completion_at_end=True, fp32_acc=True,
            )

        fi_us = _profile("flashinfer", fi_call, cuda_graph=True)
        flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(fi_handles, dist.group.WORLD)
    except Exception as e:
        if rank == 0:
            print(f"FlashInfer unavailable: {e}")

    if rank == 0:
        print(f"\n{'='*65}")
        print(f"  SUMMARY  (NUM_TOKENS={num_tokens}, HIDDEN={hidden}, world_size={world_size})")
        print(f"{'='*65}")
        print(f"  {'Variant':<25s} {'us/iter':>10s} {'vs baseline':>12s}")
        print(f"  {'-'*25} {'-'*10} {'-'*12}")
        results = [("baseline", baseline_us), ("fused_op", fused_op_us), ("lamport", lamport_us)]
        if fi_us is not None:
            results.append(("flashinfer", fi_us))
        for name, us in results:
            print(f"  {name:<25s} {us:10.1f} {baseline_us/us:11.2f}x")
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
