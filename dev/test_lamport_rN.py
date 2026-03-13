"""
Test Lamport repeated at specific ROWS.

Launch: torchrun --nproc_per_node=2 dev/test_lamport_rN.py <rows>
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

    symm_mem.get_symm_mem_workspace(
        group_name, min_size=ROWS * HIDDEN * 2
    )

    if rank == 0:
        print(f"Testing ROWS={ROWS}, HIDDEN={HIDDEN}, repeated 6x", flush=True)

    @torch.compile(options={
        "_fuse_symm_mem_comms": True,
        "_symm_mem_sync_mode": "lamport",
    })
    def fn(inp, res, w, gn):
        reduced = funcol.all_reduce(inp, "sum", group=gn)
        h = reduced + res
        normed = F.rms_norm(h, w.shape, w, EPS)
        return normed, h

    with torch.inference_mode():
        for i in range(6):
            x_new = torch.randn(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
            res_new = torch.randn(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
            expected_normed, expected_pre_norm = reference(x_new, weight, EPS, residual=res_new)
            result_normed, result_h = fn(x_new, res_new, weight, group_name)
            torch.testing.assert_close(result_normed, expected_normed, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(result_h, expected_pre_norm, atol=2e-2, rtol=2e-2)
            if rank == 0:
                print(f"  iter {i}: OK", flush=True)

    dist.destroy_process_group()
    if rank == 0:
        print("Done.", flush=True)

if __name__ == "__main__":
    main()
