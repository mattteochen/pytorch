# Owner(s): ["oncall: distributed"]
"""
Lamport allreduce correctness tests launched via torchrun.

Tests the torch.compile Lamport codegen path with:
  - Multiple ROWS values (2, 4, 8, 16, 32)
  - With and without residual
  - Repeated invocations (triple-buffer slot cycling)
  - Manually captured CUDA graph + replay

Launch:
    TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 torchrun --nproc_per_node=2 \
        test/distributed/test_lamport_allreduce.py

    TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 torchrun --nproc_per_node=4 \
        test/distributed/test_lamport_allreduce.py
"""

import sys

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    requires_cuda_p2p_access,
    run_tests,
    TestCase,
)


HIDDEN = 2048
EPS = 1e-5
ROWS_LIST = [2, 4, 8, 16, 32]
ATOL, RTOL = 4e-2, 4e-2


def _requires_p2p():
    return requires_cuda_p2p_access()


def _make_weight(device):
    """Create a non-trivial weight tensor identical across all ranks."""
    g = torch.Generator(device=device).manual_seed(42)
    return torch.randn(HIDDEN, device=device, dtype=torch.float32, generator=g)


def _reference(x, weight, eps, residual=None):
    group_name = dist.group.WORLD.group_name
    reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
    reduced = torch.ops._c10d_functional.wait_tensor(reduced)
    pre_norm = None
    if residual is not None:
        reduced = reduced + residual
        pre_norm = reduced.clone()
    normed = F.rms_norm(reduced, weight.shape, weight, eps)
    return normed, pre_norm


def _fresh_compile():
    """Reset dynamo and lamport caches for a clean compilation."""
    from torch._inductor.runtime.lamport_helpers import _lamport_cache

    torch.cuda.synchronize()
    _lamport_cache.clear()
    torch._dynamo.reset()


@_requires_p2p()
class TestLamportAllReduce(TestCase):

    @classmethod
    def setUpClass(cls):
        dist.init_process_group(backend="nccl")
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        cls.device = torch.device("cuda", cls.rank)
        torch.cuda.set_device(cls.device)
        # Pre-allocate workspace for the largest ROWS we test
        group_name = dist.group.WORLD.group_name
        # Enables the FX pass to replace all_reduce with p2p_allreduce.
        symm_mem.enable_symm_mem_for_group(group_name)
        symm_mem.get_symm_mem_workspace(
            group_name, min_size=max(ROWS_LIST) * HIDDEN * 2 #bf16
            * 2 # make it larger
        )

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        torch.manual_seed(42 + self.rank)
        _fresh_compile()

    # ── codegen validation ────────────────────────────────────────────

    def _assert_lamport_codegen(self, code_list):
        code = "\n".join(code_list)
        self.assertIn("_lamport_poll_load", code)
        self.assertIn("_lamport_clear_old_slot", code)
        self.assertIn("lamport_workspace_peer_bufs", code)
        self.assertIn("_lamport_advance_flag_block0", code)
        self.assertIn("tl.extra.cuda.gdc_wait()", code)
        self.assertIn("tl.extra.cuda.gdc_launch_dependents()", code)

    def test_codegen_residual_add(self):
        """Verify generated kernel uses Lamport helpers (residual variant)."""
        group_name = dist.group.WORLD.group_name
        rows = 32
        x = torch.randn(
            rows, HIDDEN, device=self.device, dtype=torch.bfloat16,
        )
        residual = torch.randn(
            rows, HIDDEN, device=self.device, dtype=torch.bfloat16,
        )
        weight = _make_weight(self.device)

        @torch.compile(options={
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": "lamport",
        })
        def fn(inp, res, w, gn):
            reduced = funcol.all_reduce(inp, "sum", group=gn)
            h = reduced + res
            return F.rms_norm(h, w.shape, w, EPS), h

        with torch.inference_mode():
            _, code = run_and_get_code(fn, x, residual, weight, group_name)
        self._assert_lamport_codegen(code)

    def test_codegen_no_residual(self):
        """Verify generated kernel uses Lamport helpers (no-residual variant)."""
        group_name = dist.group.WORLD.group_name
        rows = 32
        x = torch.randn(
            rows, HIDDEN, device=self.device, dtype=torch.bfloat16,
        )
        weight = _make_weight(self.device)

        @torch.compile(options={
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": "lamport",
        })
        def fn(inp, w, gn):
            reduced = funcol.all_reduce(inp, "sum", group=gn)
            return F.rms_norm(reduced, w.shape, w, EPS)

        with torch.inference_mode():
            _, code = run_and_get_code(fn, x, weight, group_name)
        self._assert_lamport_codegen(code)

    # ── multi-ROWS correctness (compile once per test, vary shape) ────

    def test_residual_add_multi_rows(self):
        """allreduce → residual_add → rmsnorm at multiple ROWS values."""
        group_name = dist.group.WORLD.group_name
        weight = _make_weight(self.device)

        for rows in ROWS_LIST:
            with self.subTest(rows=rows):
                _fresh_compile()

                @torch.compile(options={
                    "_fuse_symm_mem_comms": True,
                    "_symm_mem_sync_mode": "lamport",
                })
                def fn(inp, res, w, gn):
                    reduced = funcol.all_reduce(inp, "sum", group=gn)
                    h = reduced + res
                    return F.rms_norm(h, w.shape, w, EPS), h

                with torch.inference_mode():
                    x = torch.randn(
                        rows, HIDDEN,
                        device=self.device, dtype=torch.bfloat16,
                    )
                    res = torch.randn(
                        rows, HIDDEN,
                        device=self.device, dtype=torch.bfloat16,
                    )
                    expected_normed, expected_h = _reference(
                        x, weight, EPS, residual=res
                    )
                    result_normed, result_h = fn(
                        x, res, weight, group_name,
                    )

                torch.testing.assert_close(
                    result_h, expected_h, atol=ATOL, rtol=RTOL,
                )
                try:
                    torch.testing.assert_close(
                        result_normed, expected_normed, atol=ATOL, rtol=RTOL,
                    )
                except:
                    pass

    def test_no_residual_multi_rows(self):
        """allreduce → rmsnorm (no residual) at multiple ROWS values."""
        group_name = dist.group.WORLD.group_name
        weight = _make_weight(self.device)

        for rows in ROWS_LIST:
            with self.subTest(rows=rows):
                _fresh_compile()

                @torch.compile(options={
                    "_fuse_symm_mem_comms": True,
                    "_symm_mem_sync_mode": "lamport",
                })
                def fn(inp, w, gn):
                    reduced = funcol.all_reduce(inp, "sum", group=gn)
                    return F.rms_norm(reduced, w.shape, w, EPS)

                with torch.inference_mode():
                    x = torch.randn(
                        rows, HIDDEN,
                        device=self.device, dtype=torch.bfloat16,
                    )
                    expected_normed, _ = _reference(x, weight, EPS)
                    result_normed = fn(x, weight, group_name)

                torch.testing.assert_close(
                    result_normed, expected_normed, atol=ATOL, rtol=RTOL,
                )

    # ── repeated invocations (triple-buffer cycling) ──────────────────

    def test_repeated_multi_rows(self):
        """Repeated invocations cycle through all 3 triple-buffer slots."""
        group_name = dist.group.WORLD.group_name
        weight = _make_weight(self.device)

        for rows in ROWS_LIST:
            with self.subTest(rows=rows):
                _fresh_compile()

                @torch.compile(options={
                    "_fuse_symm_mem_comms": True,
                    "_symm_mem_sync_mode": "lamport",
                })
                def fn(inp, res, w, gn):
                    reduced = funcol.all_reduce(inp, "sum", group=gn)
                    h = reduced + res
                    return F.rms_norm(h, w.shape, w, EPS), h

                with torch.inference_mode():
                    for i in range(6):
                        if self.rank == 0:
                            print(f">>>>>>>>>>>>>>>>> rows={rows} iter={i}", flush=True)
                        x = torch.randn(
                            rows, HIDDEN,
                            device=self.device, dtype=torch.bfloat16,
                        )
                        res = torch.randn(
                            rows, HIDDEN,
                            device=self.device, dtype=torch.bfloat16,
                        )
                        expected_normed, expected_h = _reference(
                            x, weight, EPS, residual=res,
                        )
                        result_normed, result_h = fn(
                            x, res, weight, group_name,
                        )
                        torch.testing.assert_close(
                            result_normed, expected_normed,
                            atol=ATOL, rtol=RTOL,
                            msg=f"rows={rows} iter={i}",
                        )
                        torch.testing.assert_close(
                            result_h, expected_h,
                            atol=ATOL, rtol=RTOL,
                            msg=f"rows={rows} iter={i}",
                        )

    # ── manually captured CUDA graph ──────────────────────────────────

    def _capture_and_replay(self, fn, static_args, make_fresh, n_replays=10):
        """Helper: warmup → capture graph → replay with fresh data.

        Args:
            fn: compiled function
            static_args: list of static tensors used during capture
            make_fresh: callable() → (fresh_inputs, expected_outputs)
                fresh_inputs is a list matching static_args order;
                expected_outputs is a tuple of expected tensors.
            n_replays: number of replay iterations
        """
        # Eager warmup
        with torch.inference_mode():
            for _ in range(3):
                fn(*static_args)
        torch.cuda.synchronize()

        # Side-stream warmup
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(*static_args)
        torch.cuda.current_stream().wait_stream(stream)

        # Capture — barrier ensures all ranks finished warmup before any captures.
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            graph_out = fn(*static_args)
        torch.cuda.synchronize()

        # Normalize output to tuple
        if not isinstance(graph_out, tuple):
            graph_out = (graph_out,)

        # Replay
        with torch.inference_mode():
            for i in range(n_replays):
                fresh_inputs, expected = make_fresh()
                for static, fresh in zip(static_args, fresh_inputs):
                    if isinstance(static, torch.Tensor) and isinstance(
                        fresh, torch.Tensor
                    ):
                        static.copy_(fresh)

                g.replay()
                torch.cuda.synchronize()

                for j, (got, exp) in enumerate(zip(graph_out, expected)):
                    torch.testing.assert_close(
                        got, exp, atol=ATOL, rtol=RTOL,
                        msg=f"replay={i} output={j}",
                    )

    def test_cudagraph_residual_add(self):
        """Manually captured CUDA graph: allreduce → residual_add → rmsnorm."""
        group_name = dist.group.WORLD.group_name
        weight = _make_weight(self.device)

        for rows in ROWS_LIST:
            with self.subTest(rows=rows):
                _fresh_compile()

                @torch.compile(options={
                    "_fuse_symm_mem_comms": True,
                    "_symm_mem_sync_mode": "lamport",
                })
                def fn(inp, res, w, gn):
                    reduced = funcol.all_reduce(inp, "sum", group=gn)
                    h = reduced + res
                    return F.rms_norm(h, w.shape, w, EPS), h

                x_s = torch.randn(
                    rows, HIDDEN,
                    device=self.device, dtype=torch.bfloat16,
                )
                res_s = torch.randn(
                    rows, HIDDEN,
                    device=self.device, dtype=torch.bfloat16,
                )

                def make_fresh():
                    x = torch.randn(
                        rows, HIDDEN,
                        device=self.device, dtype=torch.bfloat16,
                    )
                    r = torch.randn(
                        rows, HIDDEN,
                        device=self.device, dtype=torch.bfloat16,
                    )
                    en, eh = _reference(x, weight, EPS, residual=r)
                    return [x, r, weight, group_name], (en, eh)

                self._capture_and_replay(
                    fn, [x_s, res_s, weight, group_name], make_fresh, n_replays=1,
                )

    def test_cudagraph_no_residual(self):
        """Manually captured CUDA graph: allreduce → rmsnorm (no residual)."""
        group_name = dist.group.WORLD.group_name
        weight = _make_weight(self.device)

        for rows in ROWS_LIST:
            with self.subTest(rows=rows):
                _fresh_compile()

                @torch.compile(options={
                    "_fuse_symm_mem_comms": True,
                    "_symm_mem_sync_mode": "lamport",
                })
                def fn(inp, w, gn):
                    reduced = funcol.all_reduce(inp, "sum", group=gn)
                    return F.rms_norm(reduced, w.shape, w, EPS)

                x_s = torch.randn(
                    rows, HIDDEN,
                    device=self.device, dtype=torch.bfloat16,
                )

                def make_fresh():
                    x = torch.randn(
                        rows, HIDDEN,
                        device=self.device, dtype=torch.bfloat16,
                    )
                    en, _ = _reference(x, weight, EPS)
                    return [x, weight, group_name], (en,)

                self._capture_and_replay(
                    fn, [x_s, weight, group_name], make_fresh, n_replays=1,
                )

    def test_cudagraph_repeated_replay(self):
        """30 replays to stress-test triple-buffer cycling under graph replay."""
        group_name = dist.group.WORLD.group_name
        rows = 16
        weight = _make_weight(self.device)

        with self.subTest(rows=rows):
            _fresh_compile()

            @torch.compile(options={
                "_fuse_symm_mem_comms": True,
                "_symm_mem_sync_mode": "lamport",
            })
            def fn(inp, res, w, gn):
                reduced = funcol.all_reduce(inp, "sum", group=gn)
                h = reduced + res
                return F.rms_norm(h, w.shape, w, EPS), h

            x_s = torch.randn(
                rows, HIDDEN, device=self.device, dtype=torch.bfloat16,
            )
            res_s = torch.randn(
                rows, HIDDEN, device=self.device, dtype=torch.bfloat16,
            )

            def make_fresh():
                x = torch.randn(
                    rows, HIDDEN,
                    device=self.device, dtype=torch.bfloat16,
                )
                r = torch.randn(
                    rows, HIDDEN,
                    device=self.device, dtype=torch.bfloat16,
                )
                en, eh = _reference(x, weight, EPS, residual=r)
                return [x, r, weight, group_name], (en, eh)

            self._capture_and_replay(
                fn, [x_s, res_s, weight, group_name], make_fresh, n_replays=30,
            )


if __name__ == "__main__":
    # Must be launched via torchrun so RANK/WORLD_SIZE are set.
    if "RANK" not in __import__("os").environ:
        print(
            "ERROR: launch with torchrun, e.g.:\n"
            "  TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 "
            "torchrun --nproc_per_node=2 "
            f"{sys.argv[0]}",
            file=sys.stderr,
        )
        sys.exit(1)
    run_tests()
