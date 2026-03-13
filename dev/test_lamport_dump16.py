"""
Dump generated code for ROWS=16.

Launch: torchrun --nproc_per_node=2 dev/test_lamport_dump16.py
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

HIDDEN = 2048
EPS = 1e-5
ROWS = 16


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

    x = torch.randn(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)

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
        (result_normed, result_h), code = run_and_get_code(
            fn, x, residual, weight, group_name
        )

    if rank == 0:
        for i, c in enumerate(code):
            with open(f"/tmp/lamport_code_{i}.py", "w") as f:
                f.write(c)
            print(f"Code[{i}] written to /tmp/lamport_code_{i}.py ({len(c)} chars)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
