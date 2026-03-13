"""
Test Lamport repeated at ROWS=32 WITHOUT interleaved reference calls.

Launch: torchrun --nproc_per_node=2 dev/test_lamport_no_ref.py
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F

HIDDEN = 2048
EPS = 1e-5
ROWS = 32

def reference(x, weight, eps, residual=None):
    group_name = dist.group.WORLD.group_name
    reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
    reduced = torch.ops._c10d_functional.wait_tensor(reduced)
    if residual is not None:
        reduced = reduced + residual
    normed = F.rms_norm(reduced, weight.shape, weight, eps)
    return normed, reduced

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.manual_seed(42 + rank)

    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)
    weight = torch.ones(HIDDEN, device=device, dtype=torch.float32)
    symm_mem.get_symm_mem_workspace(group_name, min_size=ROWS * HIDDEN * 2)

    @torch.compile(options={
        "_fuse_symm_mem_comms": True,
        "_symm_mem_sync_mode": "lamport",
    })
    def fn(inp, res, w, gn):
        reduced = funcol.all_reduce(inp, "sum", group=gn)
        h = reduced + res
        normed = F.rms_norm(h, w.shape, w, EPS)
        return normed, h

    if rank == 0:
        print(f"Testing ROWS={ROWS} repeated 6x (no interleaved reference)", flush=True)

    # First: run all 6 fn() calls, save results
    results = []
    inputs = []
    with torch.inference_mode():
        for i in range(6):
            x_new = torch.randn(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
            res_new = torch.randn(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
            inputs.append((x_new, res_new))
            result_normed, result_h = fn(x_new, res_new, weight, group_name)
            results.append((result_normed.clone(), result_h.clone()))
            if rank == 0:
                print(f"  fn iter {i}: OK (ran without hang)", flush=True)

    # Then: verify with reference
    for i, (x_new, res_new) in enumerate(inputs):
        expected_normed, expected_h = reference(x_new, weight, EPS, residual=res_new)
        result_normed, result_h = results[i]
        diff_h = (result_h.float() - expected_h.float()).abs().max().item()
        diff_n = (result_normed.float() - expected_normed.float()).abs().max().item()
        ok = diff_h < 4e-2 and diff_n < 4e-2
        if rank == 0:
            print(f"  check iter {i}: {'OK' if ok else 'FAIL'} h_max={diff_h:.6f} n_max={diff_n:.6f}", flush=True)

    dist.destroy_process_group()
    if rank == 0:
        print("Done.", flush=True)

if __name__ == "__main__":
    main()
