"""
Tests for multi-phase persistent reduction fusion.

Validates that a full-row reduction (e.g. RMSNorm variance) followed by a
per-group reduction (e.g. fp8 quantization amax) can be fused into a single
persistent Triton kernel.
"""

import torch
import torch._inductor.config
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON as HAS_CUDA


fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max


def rmsnorm_fp8_quant(x, residual, weight, eps, group_size):
    x = x + residual
    residual_out = x
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    x_normed = x_normed * (1.0 + weight.float())
    M, K = x_normed.shape
    x_grouped = x_normed.reshape(M, K // group_size, group_size)
    amax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = amax / fp8_max
    x_scaled = (x_grouped / scale).clamp(-fp8_max, fp8_max)
    x_q = x_scaled.to(fp8_dtype).reshape(M, K)
    x_s = scale.squeeze(-1)
    return x_q, x_s, residual_out


class TestMultiPhaseReduction(TestCase):
    @torch._inductor.config.patch(
        {
            "triton.multi_phase_persistent_reduction": True,
        }
    )
    def test_rmsnorm_fp8_quant_fusion_correctness(self):
        """Compiled output matches eager for RMSNorm + fp8 quant."""
        if not HAS_CUDA:
            self.skipTest("requires CUDA")

        M, hidden, eps, group_size = 1, 2048, 1e-6, 128

        torch.manual_seed(42)
        x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda")
        res = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")

        def fn(x, res, w):
            return rmsnorm_fp8_quant(x, res, w, eps, group_size)

        compiled = torch.compile(fn, fullgraph=True)

        q_eager, s_eager, r_eager = fn(x, res, w)
        q_comp, s_comp, r_comp = compiled(x, res, w)

        self.assertTrue(torch.equal(r_eager, r_comp))
        self.assertTrue(
            torch.allclose(s_eager, s_comp, rtol=1e-2, atol=1e-4),
            f"scale mismatch: {(s_eager - s_comp).abs().max()}",
        )
        q_match = (q_eager == q_comp).float().mean().item()
        self.assertGreater(q_match, 0.95, f"quant match {q_match:.1%}")

    @torch._inductor.config.patch(
        {
            "triton.multi_phase_persistent_reduction": True,
        }
    )
    def test_rmsnorm_fp8_quant_kernel_count(self):
        """Verify the fusion produces fewer kernels than without fusion."""
        if not HAS_CUDA:
            self.skipTest("requires CUDA")

        M, hidden, eps, group_size = 1, 2048, 1e-6, 128

        torch.manual_seed(42)
        x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda")
        res = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")

        def fn(x, res, w):
            return rmsnorm_fp8_quant(x, res, w, eps, group_size)

        from unittest.mock import patch
        from torch._inductor import metrics

        metrics.reset()
        compiled = torch.compile(fn, fullgraph=True)
        compiled(x, res, w)
        torch.cuda.synchronize()

        # With multi-phase fusion the two reductions (variance + amax),
        # the quantization epilogue, and the scale computation are all
        # fused into a single persistent kernel.
        self.assertEqual(
            metrics.generated_kernel_count,
            1,
            f"Expected 1 kernel, got {metrics.generated_kernel_count}",
        )


if __name__ == "__main__":
    run_tests()
