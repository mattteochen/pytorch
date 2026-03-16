# Owner(s): ["oncall: distributed"]
"""
Tests for fused all_reduce + rmsnorm:
  - The new P2P allreduce FX pass (replaces all_reduce+wait with p2p_allreduce)
  - The original fused_all_reduce_rmsnorm op (return-type/correctness)
  - Multi-GPU distributed correctness
"""

import operator
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from functorch import make_fx
from torch._inductor.decomposition import decompositions
from torch._inductor.fx_passes.fuse_symm_mem_comms import (
    _find_all_reduce_wait_patterns,
    _is_all_reduce,
    _is_wait_tensor,
    fuse_symm_mem_comms_pass,
)
from torch._inductor.fx_passes.post_grad import remove_noop_ops, view_to_reshape
from torch._inductor.utils import run_and_get_code
from torch.distributed._functional_collectives import all_reduce
from torch.distributed._symmetric_memory import _test_mode
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_p2p_access,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


def _make_post_grad_fx(f, *inps):
    """Create a post-grad FX graph from a function."""
    gm = make_fx(f, decompositions, tracing_mode="fake")(*inps)
    remove_noop_ops(gm.graph)
    view_to_reshape(gm)
    return gm


# ── Pattern detection tests ──────────────────────────────────────────────


class TestP2PAllReducePatternDetection(TestCase):
    """Test the new FX pass that replaces all_reduce+wait with p2p_allreduce."""

    def setUp(self):
        super().setUp()
        store = FakeStore()
        dist.init_process_group(backend="fake", world_size=2, rank=0, store=store)

    def tearDown(self):
        dist.destroy_process_group()
        super().tearDown()

    def test_find_all_reduce_wait_pattern(self):
        group = dist.group.WORLD
        hidden_dim = 64

        def func(x, weight):
            reduced = all_reduce(x, "sum", group=group.group_name)
            return F.rms_norm(reduced, (hidden_dim,), weight, eps=1e-6)

        gm = _make_post_grad_fx(
            func, torch.randn(4, hidden_dim), torch.randn(hidden_dim)
        )
        patterns = _find_all_reduce_wait_patterns(gm.graph)
        self.assertEqual(len(patterns), 1)

    def test_multiple_patterns(self):
        group = dist.group.WORLD
        hidden_dim = 64

        def func(x1, x2, weight):
            r1 = all_reduce(x1, "sum", group=group.group_name)
            o1 = F.rms_norm(r1, (hidden_dim,), weight, eps=1e-6)
            r2 = all_reduce(x2, "sum", group=group.group_name)
            o2 = F.rms_norm(r2, (hidden_dim,), weight, eps=1e-6)
            return o1, o2

        gm = _make_post_grad_fx(
            func,
            torch.randn(4, hidden_dim),
            torch.randn(4, hidden_dim),
            torch.randn(hidden_dim),
        )
        patterns = _find_all_reduce_wait_patterns(gm.graph)
        self.assertEqual(len(patterns), 2)

    def test_no_match_without_all_reduce(self):
        hidden_dim = 64

        def func(x, weight):
            return F.rms_norm(x, (hidden_dim,), weight, eps=1e-6)

        gm = _make_post_grad_fx(
            func, torch.randn(4, hidden_dim), torch.randn(hidden_dim)
        )
        patterns = _find_all_reduce_wait_patterns(gm.graph)
        self.assertEqual(len(patterns), 0)

    def test_pass_replaces_with_p2p_allreduce(self):
        """After the pass, all_reduce+wait should be replaced with p2p_allreduce."""
        group = dist.group.WORLD
        hidden_dim = 64

        def func(x, weight):
            reduced = all_reduce(x, "sum", group=group.group_name)
            return F.rms_norm(reduced, (hidden_dim,), weight, eps=1e-6)

        gm = _make_post_grad_fx(
            func, torch.randn(4, hidden_dim), torch.randn(hidden_dim)
        )

        with _test_mode():
            fuse_symm_mem_comms_pass(gm.graph)

        # p2p_allreduce should be in the graph
        p2p_nodes = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.symm_mem.p2p_allreduce.default
        ]
        self.assertEqual(len(p2p_nodes), 1, "Should have one p2p_allreduce node")

        # all_reduce and wait_tensor should be gone
        ar_nodes = [n for n in gm.graph.nodes if _is_all_reduce(n)]
        wait_nodes = [n for n in gm.graph.nodes if _is_wait_tensor(n)]
        self.assertEqual(len(ar_nodes), 0, "all_reduce should be removed")
        self.assertEqual(len(wait_nodes), 0, "wait_tensor should be removed")

    def test_pass_preserves_downstream_ops(self):
        """The RMSNorm decomposition should remain after the pass."""
        group = dist.group.WORLD
        hidden_dim = 64

        def func(x, residual, weight):
            reduced = all_reduce(x, "sum", group=group.group_name)
            h = reduced + residual
            return F.rms_norm(h, (hidden_dim,), weight, eps=1e-6)

        gm = _make_post_grad_fx(
            func,
            torch.randn(4, hidden_dim),
            torch.randn(4, hidden_dim),
            torch.randn(hidden_dim),
        )

        with _test_mode():
            fuse_symm_mem_comms_pass(gm.graph)

        # The add and rmsnorm decomposition ops should still be present
        add_nodes = [
            n
            for n in gm.graph.nodes
            if n.target in (torch.ops.aten.add.Tensor, operator.add)
        ]
        self.assertGreater(len(add_nodes), 0, "add should still be in graph")

    def test_pass_skips_training_mode(self):
        """Pass should be a no-op in training mode."""
        group = dist.group.WORLD
        hidden_dim = 64

        def func(x, weight):
            reduced = all_reduce(x, "sum", group=group.group_name)
            return F.rms_norm(reduced, (hidden_dim,), weight, eps=1e-6)

        gm = _make_post_grad_fx(
            func, torch.randn(4, hidden_dim), torch.randn(hidden_dim)
        )

        with _test_mode():
            fuse_symm_mem_comms_pass(gm.graph, is_inference=False)

        # all_reduce should still be present
        ar_nodes = [n for n in gm.graph.nodes if _is_all_reduce(n)]
        self.assertGreater(len(ar_nodes), 0)


# ── P2P allreduce op tests ──────────────────────────────────────────────


class TestP2PAllReduceOp(TestCase):
    """Test the p2p_allreduce custom op fallback."""

    def setUp(self):
        super().setUp()
        store = FakeStore()
        dist.init_process_group(backend="fake", world_size=2, rank=0, store=store)

    def tearDown(self):
        dist.destroy_process_group()
        super().tearDown()

    def test_fallback_returns_correct_shape(self):
        x = torch.randn(4, 64)
        group_name = dist.group.WORLD.group_name
        result = torch.ops.symm_mem.p2p_allreduce(x, "sum", group_name)
        self.assertEqual(result.shape, x.shape)

    def test_meta_returns_correct_shape(self):
        x = torch.randn(4, 64, device="meta")
        result = torch.ops.symm_mem.p2p_allreduce(x, "sum", "test_group")
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.device, x.device)


# ── Helper function tests ───────────────────────────────────────────────


class TestHelperFunctions(TestCase):
    """Test pattern detection helper functions."""

    def setUp(self):
        super().setUp()
        store = FakeStore()
        dist.init_process_group(backend="fake", world_size=2, rank=0, store=store)

    def tearDown(self):
        dist.destroy_process_group()
        super().tearDown()

    def test_is_all_reduce(self):
        group = dist.group.WORLD

        def func(x):
            return all_reduce(x, "sum", group=group.group_name)

        gm = make_fx(func, tracing_mode="fake")(torch.randn(4, 4))
        ar_nodes = [n for n in gm.graph.nodes if _is_all_reduce(n)]
        self.assertEqual(len(ar_nodes), 1)

    def test_is_wait_tensor(self):
        group = dist.group.WORLD

        def func(x):
            return all_reduce(x, "sum", group=group.group_name)

        gm = make_fx(func, tracing_mode="fake")(torch.randn(4, 4))
        wait_nodes = [n for n in gm.graph.nodes if _is_wait_tensor(n)]
        self.assertEqual(len(wait_nodes), 1)


# ── Original fused op tests (still valid) ───────────────────────────────


class TestFusedOpReturnType(TestCase):
    """Test the fused op's (Tensor, Tensor?) return semantics."""

    def setUp(self):
        super().setUp()
        store = FakeStore()
        dist.init_process_group(backend="fake", world_size=2, rank=0, store=store)

    def tearDown(self):
        dist.destroy_process_group()
        super().tearDown()

    def test_fallback_no_residual(self):
        x = torch.randn(4, 64)
        weight = torch.randn(64)
        group_name = dist.group.WORLD.group_name

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x,
            weight,
            "sum",
            group_name,
            eps=1e-6,
        )
        self.assertEqual(normed.shape, x.shape)
        self.assertIsNone(pre_norm)

    def test_fallback_with_residual(self):
        x = torch.randn(4, 64)
        residual = torch.randn(4, 64)
        weight = torch.randn(64)
        group_name = dist.group.WORLD.group_name

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x,
            weight,
            "sum",
            group_name,
            residual=residual,
            eps=1e-6,
        )
        self.assertEqual(normed.shape, x.shape)
        self.assertIsNotNone(pre_norm)
        self.assertEqual(pre_norm.shape, x.shape)

    def test_fallback_correctness_with_residual(self):
        x = torch.randn(4, 64)
        residual = torch.randn(4, 64)
        weight = torch.randn(64)
        group_name = dist.group.WORLD.group_name

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x,
            weight,
            "sum",
            group_name,
            residual=residual,
            eps=1e-6,
        )

        reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        expected_pre_norm = reduced + residual
        expected_normed = F.rms_norm(expected_pre_norm, weight.shape, weight, 1e-6)

        self.assertEqual(pre_norm, expected_pre_norm)
        self.assertEqual(normed, expected_normed)

    def test_fallback_correctness_no_residual(self):
        x = torch.randn(4, 64)
        weight = torch.randn(64)
        group_name = dist.group.WORLD.group_name

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x,
            weight,
            "sum",
            group_name,
            eps=1e-6,
        )

        reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        expected_normed = F.rms_norm(reduced, weight.shape, weight, 1e-6)

        self.assertIsNone(pre_norm)
        self.assertEqual(normed, expected_normed)


# ── Multi-GPU distributed tests ─────────────────────────────────────────

device_type = "cuda"
test_contexts = [nullcontext, _test_mode]


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
class TestFusedAllReduceRMSNormDistributed(MultiProcContinuousTest):
    """Multi-process tests that run real distributed allreduce + rmsnorm."""

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        torch.manual_seed(42 + self.rank)

    def _reference_allreduce_rmsnorm(self, x, weight, eps, residual=None):
        """Compute expected result via manual all_reduce + rms_norm."""
        group_name = dist.group.WORLD.group_name
        reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        pre_norm = None
        if residual is not None:
            reduced = reduced + residual
            pre_norm = reduced.clone()
        normed = F.rms_norm(reduced, weight.shape, weight, eps)
        return normed, pre_norm

    @skip_if_lt_x_gpu(2)
    @parametrize("ctx", test_contexts)
    def test_fused_op_no_residual(self, ctx):
        """Fused op (direct call) without residual."""
        self._init_process()
        hidden = 64
        x = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)
        weight = torch.ones(hidden, device=self.device, dtype=torch.float32)
        eps = 1e-5

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        symm_mem.get_symm_mem_workspace(
            group_name, min_size=x.numel() * x.element_size()
        )

        expected_normed, _ = self._reference_allreduce_rmsnorm(x, weight, eps)

        with ctx():
            normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
                x,
                weight,
                "sum",
                group_name,
                eps=eps,
            )

        self.assertIsNone(pre_norm)
        torch.testing.assert_close(normed, expected_normed, atol=1e-2, rtol=1e-2)

    @skip_if_lt_x_gpu(2)
    @parametrize("ctx", test_contexts)
    def test_fused_op_with_residual(self, ctx):
        """Fused op (direct call) with residual."""
        self._init_process()
        hidden = 64
        x = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)
        residual = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)
        weight = torch.ones(hidden, device=self.device, dtype=torch.float32)
        eps = 1e-5

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        symm_mem.get_symm_mem_workspace(
            group_name, min_size=x.numel() * x.element_size()
        )

        expected_normed, expected_pre_norm = self._reference_allreduce_rmsnorm(
            x,
            weight,
            eps,
            residual=residual,
        )

        with ctx():
            normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
                x,
                weight,
                "sum",
                group_name,
                residual=residual,
                eps=eps,
            )

        torch.testing.assert_close(normed, expected_normed, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(pre_norm, expected_pre_norm, atol=2e-2, rtol=2e-2)

    @skip_if_lt_x_gpu(2)
    @parametrize("ctx", test_contexts)
    def test_p2p_allreduce_op(self, ctx):
        """p2p_allreduce op fallback correctness."""
        self._init_process()
        hidden = 64
        x = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        expected = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
        expected = torch.ops._c10d_functional.wait_tensor(expected)

        with ctx():
            result = torch.ops.symm_mem.p2p_allreduce(x, "sum", group_name)

        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def _reference_allreduce_sum(self, x, residual=None):
        reduced = torch.ops._c10d_functional.all_reduce(x, "sum", "0")
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        if residual is not None:
            reduced = reduced + residual
        return reduced.sum(dim=-1), reduced

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_allreduce_sum(self):
        """E2E torch.compile: all_reduce -> add residual -> sum(dim=-1)."""
        self._init_process()
        hidden = 64

        x = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)
        residual = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        symm_mem.get_symm_mem_workspace(
            group_name, min_size=x.numel() * x.element_size()
        )

        expected_sum, expected_h = self._reference_allreduce_sum(x, residual=residual)

        @torch.compile(options={"_fuse_symm_mem_comms": True})
        def fn(inp, res, gn):
            reduced = all_reduce(inp, "sum", group=gn)
            h = reduced + res
            return h.sum(dim=-1), h

        with torch.inference_mode():
            result_sum, result_h = fn(x, residual, group_name)

        torch.testing.assert_close(result_sum, expected_sum, atol=0.2, rtol=0.1)
        torch.testing.assert_close(result_h, expected_h, atol=2e-2, rtol=2e-2)

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_allreduce_sum_no_residual(self):
        """E2E torch.compile: all_reduce -> sum(dim=-1) (no residual)."""
        self._init_process()
        hidden = 64

        x = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        symm_mem.get_symm_mem_workspace(
            group_name, min_size=x.numel() * x.element_size()
        )

        expected_sum, _ = self._reference_allreduce_sum(x)

        @torch.compile(options={"_fuse_symm_mem_comms": True})
        def fn(inp, gn):
            reduced = all_reduce(inp, "sum", group=gn)
            return reduced.sum(dim=-1)

        with torch.inference_mode():
            result = fn(x, group_name)

        torch.testing.assert_close(result, expected_sum, atol=0.2, rtol=0.1)


    def _assert_lamport_codegen(self, code_list):
        """Verify the generated code uses Lamport helpers, not pull model."""
        code = "\n".join(code_list)
        self.assertIn("_lamport_poll_load", code,
                       "Kernel should call _lamport_poll_load")
        self.assertIn("_lamport_clear_old_slot", code,
                       "Kernel should call _lamport_clear_old_slot (epilogue)")
        self.assertIn("lamport_workspace_peer_bufs", code,
                       "Wrapper should call lamport_workspace_peer_bufs")
        self.assertIn("_lamport_advance_flag_block0", code,
                       "Kernel should call _lamport_advance_flag_block0 (epilogue)")
        self.assertIn("tl.extra.cuda.gdc_wait()", code,
                       "Kernel should call gdc_wait() for PDL serialization")
        self.assertIn("tl.extra.cuda.gdc_launch_dependents()", code,
                       "Kernel should call gdc_launch_dependents() for PDL serialization")
        self.assertNotIn("lamport_advance_offsets", code,
                          "Wrapper should NOT call lamport_advance_offsets (in-kernel now)")
        self.assertNotIn("_symm_mem_sync", code,
                          "Kernel should NOT use device-side CAS sync")
        self.assertNotIn("symm_mem_host_barrier", code,
                          "Wrapper should NOT use host barriers")

    def _assert_device_cas_codegen(self, code_list):
        """Verify the generated code uses device-side CAS sync, not Lamport or host barriers."""
        code = "\n".join(code_list)
        self.assertIn("_symm_mem_sync", code,
                       "Kernel should use device-side CAS sync")
        self.assertIn("symm_signal_pad_ptrs", code,
                       "Kernel should receive signal pad pointers")
        self.assertNotIn("symm_mem_host_barrier", code,
                          "Wrapper should NOT use host barriers")
        self.assertNotIn("lamport_workspace_peer_bufs", code,
                          "Wrapper should NOT use Lamport setup")
        self.assertNotIn("_lamport_poll_load", code,
                          "Kernel should NOT use Lamport reduce")

    def _assert_host_barrier_codegen(self, code_list):
        """Verify the generated code uses host barriers, not device CAS or Lamport."""
        code = "\n".join(code_list)
        self.assertIn("symm_mem_host_barrier_peer_bufs", code,
                       "Wrapper should call host barrier setup")
        self.assertIn("symm_mem_host_barrier", code,
                       "Wrapper should call host barrier")
        self.assertNotIn("_symm_mem_sync", code,
                          "Kernel should NOT use device-side CAS sync")
        self.assertNotIn("lamport_workspace_peer_bufs", code,
                          "Wrapper should NOT use Lamport setup")
        self.assertNotIn("_lamport_poll_load", code,
                          "Kernel should NOT use Lamport reduce")

    def _compile_allreduce_sum_with_codegen(
        self, sync_mode, assert_codegen_fn, extra_options=None
    ):
        """all_reduce -> add residual -> sum(dim=-1), with codegen check."""
        self._init_process()
        hidden = 64

        x = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)
        residual = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        symm_mem.get_symm_mem_workspace(
            group_name, min_size=x.numel() * x.element_size()
        )

        expected_sum, expected_h = self._reference_allreduce_sum(x, residual=residual)

        compile_options = {
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": sync_mode,
        }
        if extra_options:
            compile_options.update(extra_options)

        @torch.compile(options=compile_options)
        def fn(inp, res, gn):
            reduced = all_reduce(inp, "sum", group=gn)
            h = reduced + res
            return h.sum(dim=-1), h

        with torch.inference_mode():
            (result_sum, result_h), code = run_and_get_code(
                fn, x, residual, group_name
            )

        torch.testing.assert_close(result_sum, expected_sum, atol=0.2, rtol=0.1)
        torch.testing.assert_close(result_h, expected_h, atol=4e-2, rtol=4e-2)
        assert_codegen_fn(code)

    def _compile_upstream_allreduce_sum_with_codegen(
        self, sync_mode, assert_codegen_fn, extra_options=None
    ):
        """mul -> all_reduce -> add residual -> sum(dim=-1), with codegen check."""
        self._init_process()
        hidden = 64

        x = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)
        residual = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        symm_mem.get_symm_mem_workspace(
            group_name, min_size=x.numel() * x.element_size()
        )

        upstream = x * 2.0
        expected_sum, expected_h = self._reference_allreduce_sum(
            upstream, residual=residual,
        )

        compile_options = {
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": sync_mode,
        }
        if extra_options:
            compile_options.update(extra_options)

        @torch.compile(options=compile_options)
        def fn(inp, res, gn):
            up = inp * 2.0
            reduced = all_reduce(up, "sum", group=gn)
            h = reduced + res
            return h.sum(dim=-1), h

        with torch.inference_mode():
            (result_sum, result_h), code = run_and_get_code(
                fn, x, residual, group_name
            )

        torch.testing.assert_close(result_sum, expected_sum, atol=0.2, rtol=0.1)
        torch.testing.assert_close(result_h, expected_h, atol=4e-2, rtol=4e-2)
        assert_codegen_fn(code)

    # --- Lamport ---

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_lamport_allreduce_sum(self):
        self._compile_allreduce_sum_with_codegen("lamport", self._assert_lamport_codegen)

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_lamport_upstream_allreduce_sum(self):
        self._compile_upstream_allreduce_sum_with_codegen(
            "lamport", self._assert_lamport_codegen
        )

    # --- device_cas ---

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_device_cas_allreduce_sum(self):
        self._compile_allreduce_sum_with_codegen(
            "device_cas", self._assert_device_cas_codegen
        )

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_device_cas_upstream_allreduce_sum(self):
        self._compile_upstream_allreduce_sum_with_codegen(
            "device_cas", self._assert_device_cas_codegen
        )

    # --- device_cas_2_shot ---

    def _assert_device_cas_2_shot_codegen(self, code_list):
        """Verify two-shot codegen: device-side CAS + reduce-scatter pattern."""
        self._assert_device_cas_codegen(code_list)
        code = "\n".join(code_list)
        self.assertIn("_2shot_chunk", code,
                       "Kernel should compute per-rank chunk size")
        self.assertIn("_2shot_col_mask", code,
                       "Kernel should mask columns to local rank's chunk")

    def _assert_device_cas_2_shot_grid_cap_codegen(self, code_list):
        self._assert_device_cas_2_shot_codegen(code_list)
        code = "\n".join(code_list)
        self.assertEqual(code.count("for _x_tile in range("), 3)
        self.assertEqual(
            code.count("_symm_x_base = _x_tile.to(tl.int64) * XBLOCK"), 3
        )
        self.assertNotIn(
            "_symm_x_base = tl.program_id(0).to(tl.int64) * XBLOCK", code
        )
        self.assertIn("\n    _2shot_chunk = r0_numel // SYMM_WORLD_SIZE", code)
        self.assertNotIn("\n        _2shot_chunk = r0_numel // SYMM_WORLD_SIZE", code)
        self.assertIn("\n    r0_index = tl.arange(0, R0_BLOCK)[None, :]", code)
        self.assertNotIn("\n        r0_index = tl.arange(0, R0_BLOCK)[None, :]", code)

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_device_cas_2_shot_allreduce_sum(self):
        self._compile_allreduce_sum_with_codegen(
            "device_cas_2_shot", self._assert_device_cas_2_shot_codegen
        )

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_device_cas_2_shot_grid_cap_allreduce_sum(self):
        self._compile_allreduce_sum_with_codegen(
            "device_cas_2_shot",
            self._assert_device_cas_2_shot_grid_cap_codegen,
            {"_symm_mem_grid_cap": 2},
        )

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_device_cas_2_shot_upstream_allreduce_sum(self):
        self._compile_upstream_allreduce_sum_with_codegen(
            "device_cas_2_shot", self._assert_device_cas_2_shot_codegen
        )

    # --- host_barrier ---

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_host_barrier_allreduce_sum(self):
        self._compile_allreduce_sum_with_codegen(
            "host_barrier", self._assert_host_barrier_codegen
        )

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_host_barrier_no_residual_sum(self):
        """host_barrier: all_reduce -> sum(dim=-1), no residual."""
        self._init_process()
        hidden = 64

        x = torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16)

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        symm_mem.get_symm_mem_workspace(
            group_name, min_size=x.numel() * x.element_size()
        )

        expected_sum, _ = self._reference_allreduce_sum(x)

        @torch.compile(options={
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": "host_barrier",
        })
        def fn(inp, gn):
            reduced = all_reduce(inp, "sum", group=gn)
            return reduced.sum(dim=-1)

        with torch.inference_mode():
            result, code = run_and_get_code(fn, x, group_name)

        torch.testing.assert_close(result, expected_sum, atol=0.2, rtol=0.1)
        self._assert_host_barrier_codegen(code)


@requires_cuda_p2p_access()
class TestLamportAllReduceRMSNormResidualAdd(MultiProcContinuousTest):
    """Correctness tests for the allreduce → residual_add → rmsnorm fusion
    under Lamport sync mode, including CUDA graph replay.

    The Lamport protocol uses a triple-buffered workspace with -0.0 sentinels,
    which is specifically designed to be safe under CUDA graph batched replay.

    All tests in this class use (rows=4, hidden=128) because the Lamport
    workspace is cached per-group and dimensioned on first allocation.
    """

    _ROWS = 32
    _HIDDEN = 2048
    world_size = 2

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        torch.manual_seed(42 + self.rank)

    def _reference(self, x, weight, eps, residual=None):
        group_name = dist.group.WORLD.group_name
        reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        pre_norm = None
        if residual is not None:
            reduced = reduced + residual
            pre_norm = reduced.clone()
        normed = F.rms_norm(reduced, weight.shape, weight, eps)
        return normed, pre_norm

    def _assert_lamport_codegen(self, code_list):
        code = "\n".join(code_list)
        self.assertIn("_lamport_poll_load", code)
        self.assertIn("_lamport_clear_old_slot", code)
        self.assertIn("lamport_workspace_peer_bufs", code)
        self.assertIn("_lamport_advance_flag_block0", code)

    @skip_if_lt_x_gpu(2)
    def test_lamport_allreduce_rmsnorm_residual_add(self):
        """allreduce → residual_add → rmsnorm, Lamport mode."""
        self._init_process()
        eps = 1e-5

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        x = torch.randn(
            self._ROWS, self._HIDDEN, device=self.device, dtype=torch.bfloat16,
        )
        residual = torch.randn(
            self._ROWS, self._HIDDEN, device=self.device, dtype=torch.bfloat16,
        )
        weight = torch.ones(
            self._HIDDEN, device=self.device, dtype=torch.float32,
        )

        symm_mem.get_symm_mem_workspace(
            group_name, min_size=x.numel() * x.element_size()
        )

        expected_normed, expected_pre_norm = self._reference(
            x, weight, eps, residual=residual,
        )

        @torch.compile(options={
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": "lamport",
        })
        def fn(inp, res, w, gn):
            reduced = all_reduce(inp, "sum", group=gn)
            h = reduced + res
            normed = F.rms_norm(h, w.shape, w, eps)
            return normed, h

        with torch.inference_mode():
            (result_normed, result_h), code = run_and_get_code(
                fn, x, residual, weight, group_name,
            )

        torch.testing.assert_close(
            result_normed, expected_normed, atol=4e-2, rtol=4e-2,
        )
        torch.testing.assert_close(
            result_h, expected_pre_norm, atol=4e-2, rtol=4e-2,
        )
        self._assert_lamport_codegen(code)

    @skip_if_lt_x_gpu(2)
    def test_lamport_allreduce_rmsnorm_no_residual(self):
        """allreduce → rmsnorm (no residual), Lamport mode."""
        self._init_process()
        eps = 1e-5

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        x = torch.randn(
            self._ROWS, self._HIDDEN, device=self.device, dtype=torch.bfloat16,
        )
        weight = torch.ones(
            self._HIDDEN, device=self.device, dtype=torch.float32,
        )

        symm_mem.get_symm_mem_workspace(
            group_name, min_size=x.numel() * x.element_size()
        )

        expected_normed, _ = self._reference(x, weight, eps)

        @torch.compile(options={
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": "lamport",
        })
        def fn(inp, w, gn):
            reduced = all_reduce(inp, "sum", group=gn)
            normed = F.rms_norm(reduced, w.shape, w, eps)
            return normed

        with torch.inference_mode():
            result_normed, code = run_and_get_code(
                fn, x, weight, group_name,
            )

        torch.testing.assert_close(
            result_normed, expected_normed, atol=4e-2, rtol=4e-2,
        )
        self._assert_lamport_codegen(code)

    @skip_if_lt_x_gpu(2)
    def test_lamport_allreduce_rmsnorm_repeated(self):
        """allreduce → residual_add → rmsnorm, repeated with fresh data.

        Verifies correctness across multiple invocations of the compiled
        function — each call advances the Lamport triple-buffer flag,
        testing the modulo-3 slot cycling.
        """
        self._init_process()
        eps = 1e-5

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        x = torch.randn(
            self._ROWS, self._HIDDEN, device=self.device, dtype=torch.bfloat16,
        )
        residual = torch.randn(
            self._ROWS, self._HIDDEN, device=self.device, dtype=torch.bfloat16,
        )
        weight = torch.ones(
            self._HIDDEN, device=self.device, dtype=torch.float32,
        )

        symm_mem.get_symm_mem_workspace(
            group_name, min_size=x.numel() * x.element_size()
        )

        @torch.compile(options={
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": "lamport",
        })
        def fn(inp, res, w, gn):
            reduced = all_reduce(inp, "sum", group=gn)
            h = reduced + res
            normed = F.rms_norm(h, w.shape, w, eps)
            return normed, h

        # Run 6 iterations to cycle through all 3 triple-buffer slots twice.
        with torch.inference_mode():
            for i in range(6):
                x_new = torch.randn(
                    self._ROWS, self._HIDDEN,
                    device=self.device, dtype=torch.bfloat16,
                )
                res_new = torch.randn(
                    self._ROWS, self._HIDDEN,
                    device=self.device, dtype=torch.bfloat16,
                )
                expected_normed, expected_pre_norm = self._reference(
                    x_new, weight, eps, residual=res_new,
                )
                result_normed, result_h = fn(
                    x_new, res_new, weight, group_name,
                )
                torch.testing.assert_close(
                    result_normed, expected_normed, atol=4e-2, rtol=4e-2,
                    msg=f"Mismatch on iteration {i}",
                )
                torch.testing.assert_close(
                    result_h, expected_pre_norm, atol=4e-2, rtol=4e-2,
                    msg=f"Mismatch on iteration {i}",
                )

    @skip_if_lt_x_gpu(2)
    def test_lamport_allreduce_rmsnorm_cudagraph(self):
        """allreduce → residual_add → rmsnorm under CUDA graph replay.

        Enables cudagraph to exercise graph capture and batched
        replay with the Lamport triple-buffer protocol.
        """
        import torch._inductor.config as inductor_config

        self._init_process()
        eps = 1e-5
        iters = 10

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        x = torch.randn(
            self._ROWS, self._HIDDEN, device=self.device, dtype=torch.bfloat16,
        )
        residual = torch.randn(
            self._ROWS, self._HIDDEN, device=self.device, dtype=torch.bfloat16,
        )
        weight = torch.ones(
            self._HIDDEN, device=self.device, dtype=torch.float32,
        )

        symm_mem.get_symm_mem_workspace(
            group_name, min_size=x.numel() * x.element_size()
        )

        @torch.compile(options={
            "_fuse_symm_mem_comms": True,
            "_symm_mem_sync_mode": "lamport",
            "triton.cudagraphs": True,
            "triton.cudagraph_trees": False,
        })
        def fn(inp, res, w, gn):
            reduced = all_reduce(inp, "sum", group=gn)
            h = reduced + res
            normed = F.rms_norm(h, w.shape, w, eps)
            return normed, h

        # Warmup so cudagraph trees capture the graph.
        with torch.inference_mode():
            for _ in range(3):
                fn(x, residual, weight, group_name)

        # Replay with fresh data — exercises triple-buffer cycling.
        with torch.inference_mode():
            for _ in range(iters):
                x_new = torch.randn(
                    self._ROWS, self._HIDDEN,
                    device=self.device, dtype=torch.bfloat16,
                )
                res_new = torch.randn(
                    self._ROWS, self._HIDDEN,
                    device=self.device, dtype=torch.bfloat16,
                )
                expected_normed, expected_pre_norm = self._reference(
                    x_new, weight, eps, residual=res_new,
                )
                result_normed, result_h = fn(
                    x_new, res_new, weight, group_name,
                )
                torch.testing.assert_close(
                    result_normed, expected_normed, atol=4e-2, rtol=4e-2,
                )
                torch.testing.assert_close(
                    result_h, expected_pre_norm, atol=4e-2, rtol=4e-2,
                )
if __name__ == "__main__":
    run_tests()
