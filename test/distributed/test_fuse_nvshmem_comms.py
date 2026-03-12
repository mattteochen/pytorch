# Owner(s): ["oncall: distributed"]
# To run:
# python test/distributed/test_fuse_nvshmem_comms.py
"""
Tests for fused allreduce with NVSHMEM-backed symmetric memory.

Uses the Lamport push-model (sync_mode="nvshmem") over NVSHMEM-allocated
buffers. Separated from test_fuse_symm_mem_comms.py because
set_backend("NVSHMEM") cannot coexist with the CUDA P2P backend in the
same process.
"""

import sys

from torch.testing._internal.common_utils import TEST_WITH_ROCM

if TEST_WITH_ROCM:
    print("NVSHMEM not available on ROCm, skipping tests")
    sys.exit(0)

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.distributed._functional_collectives import all_reduce
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
)
from torch.testing._internal.inductor_utils import IS_H100

if not symm_mem.is_nvshmem_available():
    print("NVSHMEM not available, skipping tests")
    sys.exit(0)


def requires_h100():
    return skip_but_pass_in_sandcastle_if(
        not IS_H100, "NVSHMEM requires H100. Skipping test on non-H100 GPU."
    )


device_type = "cuda"


class TestFusedNvshmemAllReduceDistributed(MultiProcContinuousTest):
    """Multi-process tests for Lamport allreduce over NVSHMEM-backed symm_mem.

    Must be in a separate file so set_backend("NVSHMEM") is called before
    any other symmetric memory backend is used.
    """

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        torch.manual_seed(42 + self.rank)
        symm_mem.set_backend("NVSHMEM")

    def _reference_allreduce_sum(self, x, residual=None):
        group_name = dist.group.WORLD.group_name
        reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        if residual is not None:
            reduced = reduced + residual
        return reduced.sum(dim=-1), reduced

    def _assert_lamport_codegen(self, code_list):
        """Verify generated code uses Lamport helpers over NVSHMEM-backed buffers."""
        code = "\n".join(code_list)
        self.assertIn("_lamport_poll_all_peers", code,
                       "Kernel should call _lamport_poll_all_peers")
        self.assertIn("_lamport_clear_old_slot", code,
                       "Kernel should call _lamport_clear_old_slot")
        self.assertIn("lamport_workspace_peer_bufs", code,
                       "Wrapper should call lamport_workspace_peer_bufs")
        self.assertIn("_lamport_advance_flag_block0", code,
                       "Kernel should call _lamport_advance_flag_block0")
        self.assertIn("tl.extra.cuda.gdc_wait()", code,
                       "Kernel should call gdc_wait() for PDL")
        self.assertNotIn("_symm_mem_sync", code,
                          "Kernel should NOT use device-side CAS sync")
        self.assertNotIn("symm_mem_host_barrier", code,
                          "Wrapper should NOT use host barriers")

    @skip_if_lt_x_gpu(2)
    @requires_h100()
    def test_nvshmem_allreduce_sum(self):
        """allreduce + add + sum with nvshmem sync mode."""
        self._init_process()
        hidden = 64

        x = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)
        residual = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        expected_sum, expected_h = self._reference_allreduce_sum(
            x, residual=residual,
        )

        @torch.compile(options={
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": "nvshmem",
        })
        def fn(inp, res, gn):
            reduced = all_reduce(inp, "sum", group=gn)
            h = reduced + res
            return h.sum(dim=-1), h

        with torch.inference_mode():
            (result_sum, result_h), code = run_and_get_code(
                fn, x, residual, group_name,
            )

        torch.testing.assert_close(result_sum, expected_sum, atol=0.2, rtol=0.1)
        torch.testing.assert_close(result_h, expected_h, atol=4e-2, rtol=4e-2)
        self._assert_lamport_codegen(code)

    @skip_if_lt_x_gpu(2)
    @requires_h100()
    def test_nvshmem_upstream_allreduce_sum(self):
        """upstream compute -> allreduce + add + sum with nvshmem sync mode."""
        self._init_process()
        hidden = 64

        x = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)
        residual = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        upstream = x * 2.0
        expected_sum, expected_h = self._reference_allreduce_sum(
            upstream, residual=residual,
        )

        @torch.compile(options={
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": "nvshmem",
        })
        def fn(inp, res, gn):
            up = inp * 2.0
            reduced = all_reduce(up, "sum", group=gn)
            h = reduced + res
            return h.sum(dim=-1), h

        with torch.inference_mode():
            (result_sum, result_h), code = run_and_get_code(
                fn, x, residual, group_name,
            )

        torch.testing.assert_close(result_sum, expected_sum, atol=0.2, rtol=0.1)
        torch.testing.assert_close(result_h, expected_h, atol=4e-2, rtol=4e-2)
        self._assert_lamport_codegen(code)

    @skip_if_lt_x_gpu(2)
    @requires_h100()
    def test_nvshmem_allreduce_rmsnorm(self):
        """allreduce + residual + rmsnorm with nvshmem sync mode."""
        self._init_process()
        hidden = 64

        x = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)
        residual = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)
        weight = torch.ones(hidden, device=self.device, dtype=torch.float32)
        eps = 1e-5

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        ref = torch.ops._c10d_functional.all_reduce(x.clone(), "sum", group_name)
        ref = torch.ops._c10d_functional.wait_tensor(ref)
        expected_h = ref + residual
        expected_normed = F.rms_norm(expected_h, weight.shape, weight, eps)

        @torch.compile(options={
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": "nvshmem",
        })
        def fn(inp, res, w, gn):
            reduced = all_reduce(inp, "sum", group=gn)
            h = reduced + res
            return F.rms_norm(h, w.shape, w, eps), h

        with torch.inference_mode():
            (result_normed, result_h), code = run_and_get_code(
                fn, x, residual, weight, group_name,
            )

        torch.testing.assert_close(result_h, expected_h, atol=4e-2, rtol=4e-2)
        torch.testing.assert_close(
            result_normed, expected_normed, atol=4e-2, rtol=4e-2,
        )
        self._assert_lamport_codegen(code)


if __name__ == "__main__":
    run_tests()
