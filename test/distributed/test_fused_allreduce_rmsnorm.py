# Owner(s): ["oncall: distributed"]
"""
Tests for the fused all_reduce + rmsnorm pass.
"""

import operator
from contextlib import nullcontext
from unittest import skipIf

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from functorch import make_fx
from torch._inductor.decomposition import decompositions
from torch._inductor.fx_passes.fused_allreduce_rmsnorm import (
    _find_all_reduce_rmsnorm_patterns,
    _is_all_reduce,
    _is_wait_tensor,
    fused_all_reduce_rmsnorm_pass,
)
from torch.distributed._functional_collectives import all_reduce
from torch.distributed._symmetric_memory import _test_mode
from torch._inductor.fx_passes.post_grad import remove_noop_ops, view_to_reshape
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    PLATFORM_SUPPORTS_SYMM_MEM,
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


class TestFusedAllReduceRMSNormPatternMatching(TestCase):
    """Test pattern matching without distributed setup."""

    def setUp(self):
        super().setUp()
        self.rank = 0
        self.world_size = 2

        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    def tearDown(self):
        dist.destroy_process_group()
        super().tearDown()

    def test_find_simple_all_reduce_rmsnorm_pattern(self):
        """Test: all_reduce -> wait -> rmsnorm (no residual)."""
        group = dist.group.WORLD
        hidden_dim = 64

        def func(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            reduced = all_reduce(x, "sum", group=group.group_name)
            return F.rms_norm(reduced, (hidden_dim,), weight, eps=1e-6)

        x = torch.randn(4, hidden_dim)
        weight = torch.randn(hidden_dim)

        gm = _make_post_grad_fx(func, x, weight)

        matches = _find_all_reduce_rmsnorm_patterns(gm.graph)

        self.assertEqual(len(matches), 1, "Should find exactly one pattern")

        match = matches[0]
        self.assertIsNone(match.add_node, "Should not have add node")
        self.assertIsNone(match.residual_node, "Should not have residual")
        self.assertEqual(match.reduce_op, "sum")
        self.assertEqual(match.group_name, group.group_name)
        self.assertGreater(len(match.intermediate_nodes), 0)

    def test_find_all_reduce_add_rmsnorm_pattern(self):
        """Test: all_reduce -> wait -> add -> rmsnorm (with residual)."""
        group = dist.group.WORLD
        hidden_dim = 64

        def func(
            x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
        ) -> torch.Tensor:
            reduced = all_reduce(x, "sum", group=group.group_name)
            added = reduced + residual
            return F.rms_norm(added, (hidden_dim,), weight, eps=1e-6)

        x = torch.randn(4, hidden_dim)
        residual = torch.randn(4, hidden_dim)
        weight = torch.randn(hidden_dim)

        gm = _make_post_grad_fx(func, x, residual, weight)

        matches = _find_all_reduce_rmsnorm_patterns(gm.graph)

        self.assertEqual(len(matches), 1, "Should find exactly one pattern")

        match = matches[0]
        self.assertIsNotNone(match.add_node, "Should have add node")
        self.assertIsNotNone(match.residual_node, "Should have residual")
        self.assertEqual(match.reduce_op, "sum")

    def test_find_residual_add_reversed_order(self):
        """Test: all_reduce -> wait, then residual + wait (add order reversed)."""
        group = dist.group.WORLD
        hidden_dim = 64

        def func(
            x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
        ) -> torch.Tensor:
            reduced = all_reduce(x, "sum", group=group.group_name)
            added = residual + reduced
            return F.rms_norm(added, (hidden_dim,), weight, eps=1e-6)

        x = torch.randn(4, hidden_dim)
        residual = torch.randn(4, hidden_dim)
        weight = torch.randn(hidden_dim)

        gm = _make_post_grad_fx(func, x, residual, weight)

        matches = _find_all_reduce_rmsnorm_patterns(gm.graph)

        self.assertEqual(len(matches), 1, "Should find pattern with reversed add order")
        self.assertIsNotNone(matches[0].residual_node)

    def test_no_match_without_all_reduce(self):
        """Test: rmsnorm without all_reduce should not match."""
        hidden_dim = 64

        def func(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return F.rms_norm(x, (hidden_dim,), weight, eps=1e-6)

        x = torch.randn(4, hidden_dim)
        weight = torch.randn(hidden_dim)

        gm = _make_post_grad_fx(func, x, weight)
        matches = _find_all_reduce_rmsnorm_patterns(gm.graph)

        self.assertEqual(len(matches), 0, "Should not find any patterns")

    def test_no_match_all_reduce_without_rmsnorm(self):
        """Test: all_reduce without rmsnorm should not match."""
        group = dist.group.WORLD

        def func(x: torch.Tensor) -> torch.Tensor:
            reduced = all_reduce(x, "sum", group=group.group_name)
            return reduced * 2

        x = torch.randn(4, 64)

        gm = _make_post_grad_fx(func, x)
        matches = _find_all_reduce_rmsnorm_patterns(gm.graph)

        self.assertEqual(len(matches), 0, "Should not find any patterns")

    def test_multiple_patterns(self):
        """Test: multiple all_reduce -> rmsnorm patterns in same graph."""
        group = dist.group.WORLD
        hidden_dim = 64

        def func(
            x1: torch.Tensor, x2: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduced1 = all_reduce(x1, "sum", group=group.group_name)
            out1 = F.rms_norm(reduced1, (hidden_dim,), weight, eps=1e-6)

            reduced2 = all_reduce(x2, "sum", group=group.group_name)
            out2 = F.rms_norm(reduced2, (hidden_dim,), weight, eps=1e-6)

            return out1, out2

        x1 = torch.randn(4, hidden_dim)
        x2 = torch.randn(4, hidden_dim)
        weight = torch.randn(hidden_dim)

        gm = _make_post_grad_fx(func, x1, x2, weight)

        matches = _find_all_reduce_rmsnorm_patterns(gm.graph)

        self.assertEqual(len(matches), 2, "Should find two patterns")

    def test_avg_reduce_op(self):
        """Test: all_reduce with 'avg' reduce_op."""
        group = dist.group.WORLD
        hidden_dim = 64

        def func(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            reduced = all_reduce(x, "avg", group=group.group_name)
            return F.rms_norm(reduced, (hidden_dim,), weight, eps=1e-6)

        x = torch.randn(4, hidden_dim)
        weight = torch.randn(hidden_dim)

        gm = _make_post_grad_fx(func, x, weight)
        matches = _find_all_reduce_rmsnorm_patterns(gm.graph)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].reduce_op, "avg")


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
        """Fallback returns (normed, None) when no residual is given."""
        x = torch.randn(4, 64)
        weight = torch.randn(64)
        group_name = dist.group.WORLD.group_name

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x, weight, "sum", group_name, eps=1e-6,
        )
        self.assertEqual(normed.shape, x.shape)
        self.assertIsNone(pre_norm)

    def test_fallback_with_residual(self):
        """Fallback returns (normed, pre_norm) when residual is given."""
        x = torch.randn(4, 64)
        residual = torch.randn(4, 64)
        weight = torch.randn(64)
        group_name = dist.group.WORLD.group_name

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x, weight, "sum", group_name, residual=residual, eps=1e-6,
        )
        self.assertEqual(normed.shape, x.shape)
        self.assertIsNotNone(pre_norm)
        self.assertEqual(pre_norm.shape, x.shape)

    def test_fallback_correctness_with_residual(self):
        """pre_norm output matches manual all_reduce + residual."""
        x = torch.randn(4, 64)
        residual = torch.randn(4, 64)
        weight = torch.randn(64)
        group_name = dist.group.WORLD.group_name

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x, weight, "sum", group_name, residual=residual, eps=1e-6,
        )

        reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        expected_pre_norm = reduced + residual
        expected_normed = F.rms_norm(expected_pre_norm, weight.shape, weight, 1e-6)

        self.assertEqual(pre_norm, expected_pre_norm)
        self.assertEqual(normed, expected_normed)

    def test_fallback_correctness_no_residual(self):
        """Normed output matches manual all_reduce + rms_norm."""
        x = torch.randn(4, 64)
        weight = torch.randn(64)
        group_name = dist.group.WORLD.group_name

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x, weight, "sum", group_name, eps=1e-6,
        )

        reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        expected_normed = F.rms_norm(reduced, weight.shape, weight, 1e-6)

        self.assertIsNone(pre_norm)
        self.assertEqual(normed, expected_normed)


class TestFXPassGraphRewrite(TestCase):
    """Test that the pass correctly rewrites the FX graph with getitem nodes."""

    def setUp(self):
        super().setUp()
        store = FakeStore()
        dist.init_process_group(backend="fake", world_size=2, rank=0, store=store)

    def tearDown(self):
        dist.destroy_process_group()
        super().tearDown()

    def _count_fused_ops(self, graph):
        return sum(
            1
            for n in graph.nodes
            if n.target is torch.ops.symm_mem.fused_all_reduce_rmsnorm.default
        )

    def _count_getitems(self, graph):
        return sum(
            1 for n in graph.nodes if n.target is operator.getitem
        )

    def test_pass_inserts_getitem_nodes(self):
        """After the pass, the fused op should be followed by getitem(0) and getitem(1)."""
        group = dist.group.WORLD
        hidden_dim = 64

        def func(x, weight):
            reduced = all_reduce(x, "sum", group=group.group_name)
            return F.rms_norm(reduced, (hidden_dim,), weight, eps=1e-6)

        gm = _make_post_grad_fx(func, torch.randn(4, hidden_dim), torch.randn(hidden_dim))
        with _test_mode():
            fused_all_reduce_rmsnorm_pass(gm.graph)

        self.assertEqual(self._count_fused_ops(gm.graph), 1)
        self.assertEqual(self._count_getitems(gm.graph), 2)


class TestHelperFunctions(TestCase):
    """Test helper functions without distributed setup."""

    def setUp(self):
        super().setUp()
        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            world_size=2,
            rank=0,
            store=store,
        )

    def tearDown(self):
        dist.destroy_process_group()
        super().tearDown()

    def test_is_all_reduce(self):
        group = dist.group.WORLD

        def func(x):
            return all_reduce(x, "sum", group=group.group_name)

        gm = make_fx(func, tracing_mode="fake")(torch.randn(4, 4))

        all_reduce_nodes = [n for n in gm.graph.nodes if _is_all_reduce(n)]
        self.assertEqual(len(all_reduce_nodes), 1)

    def test_is_wait_tensor(self):
        group = dist.group.WORLD

        def func(x):
            return all_reduce(x, "sum", group=group.group_name)

        gm = make_fx(func, tracing_mode="fake")(torch.randn(4, 4))

        wait_nodes = [n for n in gm.graph.nodes if _is_wait_tensor(n)]
        self.assertEqual(len(wait_nodes), 1)


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

    @skipIf(not PLATFORM_SUPPORTS_SYMM_MEM, "Platform does not support symmetric memory")
    @skip_if_lt_x_gpu(2)
    @parametrize("context", test_contexts)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_correctness_no_residual(self, context, dtype):
        self._init_process()
        M, N = 8, 128
        eps = 1e-6
        group_name = dist.group.WORLD.group_name

        x = torch.randn(M, N, device=self.device, dtype=dtype)
        weight = torch.randn(N, device=self.device, dtype=torch.float32)

        expected_normed, _ = self._reference_allreduce_rmsnorm(x, weight, eps)

        with context():
            normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
                x, weight, "sum", group_name, eps=eps,
            )

        tol = {"rtol": 5e-2, "atol": 5e-2} if dtype == torch.bfloat16 else {"rtol": 1e-3, "atol": 1e-3}
        self.assertIsNone(pre_norm)
        torch.testing.assert_close(normed, expected_normed, **tol)

    @skipIf(not PLATFORM_SUPPORTS_SYMM_MEM, "Platform does not support symmetric memory")
    @skip_if_lt_x_gpu(2)
    @parametrize("context", test_contexts)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_correctness_with_residual(self, context, dtype):
        self._init_process()
        M, N = 8, 128
        eps = 1e-6
        group_name = dist.group.WORLD.group_name

        x = torch.randn(M, N, device=self.device, dtype=dtype)
        residual = torch.randn(M, N, device=self.device, dtype=dtype)
        weight = torch.randn(N, device=self.device, dtype=torch.float32)

        expected_normed, expected_pre_norm = self._reference_allreduce_rmsnorm(
            x, weight, eps, residual=residual,
        )

        with context():
            normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
                x, weight, "sum", group_name, residual=residual, eps=eps,
            )

        tol = {"rtol": 5e-2, "atol": 5e-2} if dtype == torch.bfloat16 else {"rtol": 1e-3, "atol": 1e-3}
        self.assertIsNotNone(pre_norm)
        torch.testing.assert_close(pre_norm, expected_pre_norm, **tol)
        torch.testing.assert_close(normed, expected_normed, **tol)

    @skipIf(not PLATFORM_SUPPORTS_SYMM_MEM, "Platform does not support symmetric memory")
    @skip_if_lt_x_gpu(2)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_3d_input(self, dtype):
        """Verify the op handles batched (3D) inputs correctly."""
        self._init_process()
        B, M, N = 2, 4, 64
        eps = 1e-6
        group_name = dist.group.WORLD.group_name

        x = torch.randn(B, M, N, device=self.device, dtype=dtype)
        weight = torch.randn(N, device=self.device, dtype=torch.float32)

        expected_normed, _ = self._reference_allreduce_rmsnorm(x, weight, eps)

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x, weight, "sum", group_name, eps=eps,
        )

        tol = {"rtol": 5e-2, "atol": 5e-2} if dtype == torch.bfloat16 else {"rtol": 1e-3, "atol": 1e-3}
        self.assertIsNone(pre_norm)
        self.assertEqual(normed.shape, x.shape)
        torch.testing.assert_close(normed, expected_normed, **tol)

    @skipIf(not PLATFORM_SUPPORTS_SYMM_MEM, "Platform does not support symmetric memory")
    @skip_if_lt_x_gpu(2)
    def test_per_rank_data_differs(self):
        """Each rank contributes different data; the reduction is real."""
        self._init_process()
        M, N = 4, 64
        eps = 1e-6
        group_name = dist.group.WORLD.group_name

        x = torch.full(
            (M, N), float(self.rank + 1), device=self.device, dtype=torch.float32
        )
        weight = torch.ones(N, device=self.device, dtype=torch.float32)

        normed, _ = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x, weight, "sum", group_name, eps=eps,
        )

        expected_sum = sum(range(1, self.world_size + 1))
        expected_reduced = torch.full(
            (M, N), float(expected_sum), device=self.device, dtype=torch.float32
        )
        expected_normed = F.rms_norm(expected_reduced, weight.shape, weight, eps)

        torch.testing.assert_close(normed, expected_normed, rtol=1e-3, atol=1e-3)


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
class TestFusedAllReduceRMSNormSymmMem(MultiProcContinuousTest):
    """Multi-process tests for the symmetric memory P2P fused kernel."""

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _init_device(self):
        torch.cuda.set_device(self.device)
        torch.manual_seed(42 + self.rank)

    def _reference_allreduce_rmsnorm(self, x, weight, eps, residual=None):
        group_name = dist.group.WORLD.group_name
        reduced = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        pre_norm = None
        if residual is not None:
            reduced = reduced + residual
            pre_norm = reduced.clone()
        normed = F.rms_norm(reduced, weight.shape, weight, eps)
        return normed, pre_norm

    @skipIf(not PLATFORM_SUPPORTS_SYMM_MEM, "Platform does not support symmetric memory")
    @skip_if_lt_x_gpu(2)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_symm_mem_no_residual(self, dtype):
        self._init_device()
        M, N = 8, 128
        eps = 1e-6
        group_name = dist.group.WORLD.group_name

        x = torch.randn(M, N, device=self.device, dtype=dtype)
        weight = torch.randn(N, device=self.device, dtype=torch.float32)

        expected_normed, _ = self._reference_allreduce_rmsnorm(x, weight, eps)

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x, weight, "sum", group_name, eps=eps,
        )

        self.assertIsNone(pre_norm)
        torch.testing.assert_close(normed, expected_normed, rtol=1e-2, atol=1e-2)

    @skipIf(not PLATFORM_SUPPORTS_SYMM_MEM, "Platform does not support symmetric memory")
    @skip_if_lt_x_gpu(2)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_symm_mem_with_residual(self, dtype):
        self._init_device()
        M, N = 8, 128
        eps = 1e-6
        group_name = dist.group.WORLD.group_name

        x = torch.randn(M, N, device=self.device, dtype=dtype)
        residual = torch.randn(M, N, device=self.device, dtype=dtype)
        weight = torch.randn(N, device=self.device, dtype=torch.float32)

        expected_normed, expected_pre_norm = self._reference_allreduce_rmsnorm(
            x, weight, eps, residual=residual,
        )

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x, weight, "sum", group_name, residual=residual, eps=eps,
        )

        self.assertIsNotNone(pre_norm)
        torch.testing.assert_close(pre_norm, expected_pre_norm, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(normed, expected_normed, rtol=1e-2, atol=1e-2)

    @skipIf(not PLATFORM_SUPPORTS_SYMM_MEM, "Platform does not support symmetric memory")
    @skip_if_lt_x_gpu(2)
    def test_symm_mem_per_rank_data_differs(self):
        self._init_device()
        M, N = 4, 64
        eps = 1e-6
        group_name = dist.group.WORLD.group_name

        x = torch.full(
            (M, N), float(self.rank + 1), device=self.device, dtype=torch.float32
        )
        weight = torch.ones(N, device=self.device, dtype=torch.float32)

        normed, _ = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x, weight, "sum", group_name, eps=eps,
        )

        expected_sum = sum(range(1, self.world_size + 1))
        expected_reduced = torch.full(
            (M, N), float(expected_sum), device=self.device, dtype=torch.float32
        )
        expected_normed = F.rms_norm(expected_reduced, weight.shape, weight, eps)

        torch.testing.assert_close(normed, expected_normed, rtol=1e-3, atol=1e-3)

    @skipIf(not PLATFORM_SUPPORTS_SYMM_MEM, "Platform does not support symmetric memory")
    @skip_if_lt_x_gpu(2)
    def test_symm_mem_3d_input(self):
        """Verify the fused kernel handles batched (3D) inputs correctly."""
        self._init_device()
        B, M, N = 2, 4, 64
        eps = 1e-6
        group_name = dist.group.WORLD.group_name

        x = torch.randn(B, M, N, device=self.device, dtype=torch.bfloat16)
        weight = torch.randn(N, device=self.device, dtype=torch.float32)

        expected_normed, _ = self._reference_allreduce_rmsnorm(x, weight, eps)

        normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
            x, weight, "sum", group_name, eps=eps,
        )

        self.assertIsNone(pre_norm)
        self.assertEqual(normed.shape, x.shape)
        torch.testing.assert_close(normed, expected_normed, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    run_tests()
