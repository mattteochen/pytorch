"""
Test Lamport with CUDA graphs at different ROWS.

Launch: torchrun --nproc_per_node=2 dev/test_lamport_cudagraph.py <rows>
"""

import sys
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F

HIDDEN = 2048
EPS = 1e-5
ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2

def reference(x, weight, eps, residual=None):
    group_name = dist.group.WORLD.group_name
    reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
    reduced = torch.ops._c10d_functional.wait_tensor(reduced)
    if residual is not None:
        reduced = reduced + residual
    normed = F.rms_norm(reduced, weight.shape, weight, eps)
    return normed, reduced if residual is not None else None

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.manual_seed(42 + rank)

    group_name = dist.group.WORLD.group_name
    weight = torch.ones(HIDDEN, device=device, dtype=torch.float32)
    symm_mem.get_symm_mem_workspace(group_name, min_size=ROWS * HIDDEN * 2)

    if rank == 0:
        print(f"Testing CUDA graph: ROWS={ROWS}, HIDDEN={HIDDEN}", flush=True)

    @torch.compile(options={
        "_fuse_symm_mem_comms": True,
        "_symm_mem_sync_mode": "lamport",
        "triton.cudagraphs": True,
        "triton.cudagraph_trees": False,
    })
    def fn(inp, res, w, gn):
        reduced = funcol.all_reduce(inp, "sum", group=gn)
        h = reduced + res
        normed = F.rms_norm(h, w.shape, w, EPS)
        return normed, h

    x = torch.randn(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)

    with torch.inference_mode():
        # Warmup (triggers compilation + cudagraph capture)
        for _ in range(3):
            fn(x, residual, weight, group_name)
        if rank == 0:
            print("  Warmup complete", flush=True)

        # Replay with fresh data
        for i in range(10):
            x_new = torch.randn(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
            res_new = torch.randn(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
            expected_normed, expected_pre_norm = reference(x_new, weight, EPS, residual=res_new)
            result_normed, result_h = fn(x_new, res_new, weight, group_name)

            diff_h = (result_h.float() - expected_pre_norm.float()).abs()
            diff_n = (result_normed.float() - expected_normed.float()).abs()
            ok = diff_h.max().item() < 2e-2 and diff_n.max().item() < 2e-2

            if rank == 0:
                print(f"  iter {i}: {'OK' if ok else 'FAIL'}  "
                      f"h_max={diff_h.max().item():.6f}  n_max={diff_n.max().item():.6f}",
                      flush=True)

    dist.destroy_process_group()
    if rank == 0:
        print("Done.", flush=True)

if __name__ == "__main__":
    main()
