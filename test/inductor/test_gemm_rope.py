# Owner(s): ["module: inductor"]
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class PackedQKVGemmRope(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        apply_rope: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.apply_rope = apply_rope

        total_qkv = self.q_size + 2 * self.kv_size
        self.qkv_weight = nn.Parameter(
            torch.randn(total_qkv, hidden_size, dtype=dtype, device=device)
        )
        self.qkv_bias = nn.Parameter(
            torch.randn(total_qkv, dtype=dtype, device=device)
        )

    def _rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x = x.view(x.shape[0], -1, self.head_dim)
        half = self.rotary_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half : 2 * half]
        out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return out.reshape(orig_shape)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = F.linear(hidden_states, self.qkv_weight, self.qkv_bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if not self.apply_rope:
            return q, k, v

        cos_sin = cos_sin_cache[positions]
        half = self.rotary_dim // 2
        cos = cos_sin[..., :half].unsqueeze(1).to(hidden_states.dtype)
        sin = cos_sin[..., half : 2 * half].unsqueeze(1).to(hidden_states.dtype)
        return self._rope(q, cos, sin), self._rope(k, cos, sin), v


@unittest.skipUnless(HAS_GPU, "requires GPU")
class GemmRopeTest(TestCase):
    device = GPU_TYPE

    def _make_model(
        self,
        *,
        dtype: torch.dtype,
        apply_rope: bool = True,
    ) -> PackedQKVGemmRope:
        model = PackedQKVGemmRope(
            hidden_size=128,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            dtype=dtype,
            device=self.device,
            apply_rope=apply_rope,
        ).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    def _run_inference_and_get_code(
        self,
        model: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ):
        with torch.inference_mode():
            with torch._inductor.config.patch(fx_graph_cache=False):
                compiled = torch.compile(model, fullgraph=True)
            return run_and_get_code(compiled, positions, hidden_states, cos_sin_cache)

    @skipIfXpu(msg="gemm_rope Triton template is CUDA-only")
    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(gemm_rope_pass=True)
    def test_gemm_rope_fuses_and_matches_eager(self):
        for dtype in (torch.bfloat16, torch.float16):
            for num_tokens in (1, 32):
                model = self._make_model(dtype=dtype)
                positions = torch.arange(
                    num_tokens, device=self.device, dtype=torch.int64
                )
                hidden_states = torch.randn(
                    num_tokens, 128, device=self.device, dtype=dtype
                )
                cos_sin_cache = torch.randn(
                    128, 8, device=self.device, dtype=torch.float32
                )

                atol = 0.125 if dtype == torch.bfloat16 else 0.02
                rtol = 0.125 if dtype == torch.bfloat16 else 0.02

                with self.subTest(dtype=dtype, num_tokens=num_tokens):
                    actual, (code,) = self._run_inference_and_get_code(
                        model, positions, hidden_states, cos_sin_cache
                    )
                    expected = model(positions, hidden_states, cos_sin_cache)

                    self.assertIn("GEMM_ROPE_TRITON_ENTRANCE", code)
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)

    @skipIfXpu(msg="gemm_rope Triton template is CUDA-only")
    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(gemm_rope_pass=True)
    def test_gemm_rope_does_not_match_without_rope(self):
        model = self._make_model(dtype=torch.bfloat16, apply_rope=False)
        positions = torch.arange(4, device=self.device, dtype=torch.int64)
        hidden_states = torch.randn(4, 128, device=self.device, dtype=torch.bfloat16)
        cos_sin_cache = torch.randn(128, 8, device=self.device, dtype=torch.float32)

        actual, (code,) = self._run_inference_and_get_code(
            model, positions, hidden_states, cos_sin_cache
        )
        expected = model(positions, hidden_states, cos_sin_cache)

        self.assertNotIn("GEMM_ROPE_TRITON_ENTRANCE", code)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want)


if __name__ == "__main__":
    run_tests()
