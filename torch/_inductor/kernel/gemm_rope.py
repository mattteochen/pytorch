# mypy: allow-untyped-defs
from dataclasses import dataclass
import functools

import torch

from ..ir import FixedLayout, TensorBox
from ..lowering import empty_strided, lowerings
from .gemm_plugin import register_gemm_plugin
from ..select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
    TritonTemplate,
    TritonTemplateCaller,
)


aten = torch.ops.aten


@dataclass(frozen=True)
class GemmTemplatePlugin:
    name: str
    extra_inputs: tuple[str, ...]
    mutated_inputs: tuple[str, ...]
    body: str


gemm_rope_configs = [
    {"BLOCK_M": bm, "BLOCK_K": bk, "num_stages": ns, "num_warps": nw}
    for bm, bk, ns, nw in [
        (16, 64, 3, 4),
        (16, 128, 5, 4),
        (32, 64, 3, 4),
        (32, 128, 3, 4),
        (64, 64, 3, 4),
        (64, 128, 2, 4),
        (128, 64, 2, 8),
        (128, 128, 2, 8),
    ]
]


@SymbolicGridFn
def gemm_rope_grid(M, N, meta, *, cdiv):
    return (cdiv(M, meta["BLOCK_M"]) * cdiv(N, meta["BLOCK_N"]), 1, 1)


def _make_gemm_template(plugin: GemmTemplatePlugin) -> TritonTemplate:
    extra_kernel_args = ", ".join(f'"{name}"' for name in plugin.extra_inputs)
    def_kernel_args = '"A", "B"'
    if extra_kernel_args:
        def_kernel_args = f'{def_kernel_args}, {extra_kernel_args}'
    return TritonTemplate(
        name=plugin.name,
        grid=gemm_rope_grid,
        debug=False,
        source=rf"""
{{{{def_kernel({def_kernel_args})}}}}


    # GEMM_ROPE_TRITON_ENTRANCE

    M = {{{{size("A", 0)}}}}
    K = {{{{size("A", 1)}}}}
    N = Q_SIZE + 2 * KV_SIZE

    stride_am = {{{{stride("A", 0)}}}}
    stride_ak = {{{{stride("A", 1)}}}}
    stride_bn = {{{{stride("B", 0)}}}}
    stride_bk = {{{{stride("B", 1)}}}}

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + rn[None, :] * stride_bn + offs_k[:, None] * stride_bk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k_idx * BLOCK_K
            a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < k_remaining), other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :] & (offs_k[:, None] < k_remaining), other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

{plugin.body}
""",
    )


def _make_rope_plugin_body(*, with_kv_cache: bool) -> str:
    cache_decl = ""
    cache_k_store = ""
    cache_v_store = ""
    if with_kv_cache:
        cache_decl = """
    stride_ckm = {{stride("K_CACHE", 0)}}
    stride_ckn = {{stride("K_CACHE", 1)}}
    stride_cvm = {{stride("V_CACHE", 0)}}
    stride_cvn = {{stride("V_CACHE", 1)}}
"""
        cache_k_store = """
            cache_rows = {{size("K_CACHE", 0)}}
            cache_row = tl.load(CACHE_LOC + rm, mask=mask_m, other=0)
            cache_row = tl.where(cache_row < 0, cache_row + cache_rows, cache_row)
            cache_mask = mask_m & (cache_row >= 0) & (cache_row < cache_rows)
            cache_row = tl.where(cache_mask, cache_row, 0)
            tl.store(
                K_CACHE + cache_row[:, None] * stride_ckm + rk[None, :] * stride_ckn,
                acc,
                mask=cache_mask[:, None] & (rk[None, :] < KV_SIZE),
            )
"""
        cache_v_store = """
        cache_rows = {{size("V_CACHE", 0)}}
        cache_row = tl.load(CACHE_LOC + rm, mask=mask_m, other=0)
        cache_row = tl.where(cache_row < 0, cache_row + cache_rows, cache_row)
        cache_mask = mask_m & (cache_row >= 0) & (cache_row < cache_rows)
        cache_row = tl.where(cache_mask, cache_row, 0)
        tl.store(
            V_CACHE + cache_row[:, None] * stride_cvm + rv[None, :] * stride_cvn,
            acc,
            mask=cache_mask[:, None] & (rv[None, :] < KV_SIZE),
        )
"""

    return (
        """
    stride_cs = {{stride("COS_SIN", 0)}}
    stride_qm = {{stride(None, 0)}}
    stride_qn = {{stride(None, 1)}}
    stride_km = {{stride("K_BUF", 0)}}
    stride_kn = {{stride("K_BUF", 1)}}
    stride_vm = {{stride("V_BUF", 0)}}
    stride_vn = {{stride("V_BUF", 1)}}
"""
        + cache_decl
        + """

    bias = tl.load(BIAS + rn, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    col_start = pid_n * BLOCK_N
    if col_start < (Q_SIZE + KV_SIZE):
        pos = tl.load(POS + rm, mask=mask_m, other=0)
        cos = tl.load(
            COS_SIN + pos[:, None] * stride_cs + tl.arange(0, HALF_ROTARY)[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        sin = tl.load(
            COS_SIN
            + pos[:, None] * stride_cs
            + HALF_ROTARY
            + tl.arange(0, HALF_ROTARY)[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        acc_3d = tl.reshape(acc, (BLOCK_M, 2, HALF_ROTARY))
        acc_t = tl.permute(acc_3d, (0, 2, 1))
        x1, x2 = tl.split(acc_t)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        out_joined = tl.join(o1, o2)
        out_t = tl.permute(out_joined, (0, 2, 1))
        acc = tl.reshape(out_t, (BLOCK_M, BLOCK_N))

        if col_start < Q_SIZE:
            tl.store(
                {{output_ptr()}} + rm[:, None] * stride_qm + rn[None, :] * stride_qn,
                acc,
                mask=mask_m[:, None] & (rn[None, :] < Q_SIZE),
            )
        else:
            rk = rn - Q_SIZE
            tl.store(
                K_BUF + rm[:, None] * stride_km + rk[None, :] * stride_kn,
                acc,
                mask=mask_m[:, None] & (rk[None, :] < KV_SIZE),
            )
"""
        + cache_k_store
        + """
    elif col_start >= (Q_SIZE + KV_SIZE):
        rv = rn - Q_SIZE - KV_SIZE
        tl.store(
            V_BUF + rm[:, None] * stride_vm + rv[None, :] * stride_vn,
            acc,
            mask=mask_m[:, None] & (rv[None, :] < KV_SIZE),
        )
"""
        + cache_v_store
    )


rope_plugin = GemmTemplatePlugin(
    name="gemm_rope",
    extra_inputs=("BIAS", "COS_SIN", "POS", "K_BUF", "V_BUF"),
    mutated_inputs=("K_BUF", "V_BUF"),
    body=_make_rope_plugin_body(with_kv_cache=False),
)


rope_kv_cache_plugin = GemmTemplatePlugin(
    name="gemm_rope_kv_cache",
    extra_inputs=(
        "BIAS",
        "COS_SIN",
        "POS",
        "K_BUF",
        "V_BUF",
        "CACHE_LOC",
        "K_CACHE",
        "V_CACHE",
    ),
    mutated_inputs=("K_BUF", "V_BUF", "K_CACHE", "V_CACHE"),
    body=_make_rope_plugin_body(with_kv_cache=True),
)


gemm_rope_template = _make_gemm_template(rope_plugin)
gemm_rope_kv_cache_template = _make_gemm_template(rope_kv_cache_plugin)


def _fallback_rope(
    x: TensorBox,
    cos: TensorBox,
    sin: TensorBox,
    head_dim: int,
    rotary_dim: int,
) -> TensorBox:
    x_view = lowerings[aten.view.default](x, [x.get_size()[0], -1, head_dim])
    half = rotary_dim // 2
    x1 = lowerings[aten.slice.Tensor](x_view, 2, 0, half)
    x2 = lowerings[aten.slice.Tensor](x_view, 2, half, rotary_dim)
    left = lowerings[aten.sub.Tensor](
        lowerings[aten.mul.Tensor](x1, cos),
        lowerings[aten.mul.Tensor](x2, sin),
    )
    right = lowerings[aten.add.Tensor](
        lowerings[aten.mul.Tensor](x2, cos),
        lowerings[aten.mul.Tensor](x1, sin),
    )
    return lowerings[aten.view.default](
        lowerings[aten.cat.default]([left, right], -1),
        list(x.get_size()),
    )


def fallback_gemm_rope(
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int,
    hidden: TensorBox,
    weight: TensorBox,
    bias: TensorBox,
    cos_sin_cache: TensorBox,
    positions: TensorBox,
):
    weight_t = lowerings[aten.t.default](weight)
    qkv = lowerings[aten.addmm.default](bias, hidden, weight_t)
    q, k, v = lowerings[aten.split_with_sizes.default](qkv, [q_size, kv_size, kv_size], -1)
    cos_sin = lowerings[aten.index.Tensor](cos_sin_cache, [positions])
    half = rotary_dim // 2
    cos = lowerings[aten.unsqueeze.default](
        lowerings[aten.slice.Tensor](cos_sin, 1, 0, half),
        1,
    )
    sin = lowerings[aten.unsqueeze.default](
        lowerings[aten.slice.Tensor](cos_sin, 1, half, rotary_dim),
        1,
    )
    return (
        _fallback_rope(q, cos, sin, head_dim, rotary_dim),
        _fallback_rope(k, cos, sin, head_dim, rotary_dim),
        v,
    )


def fallback_gemm_rope_kv_cache(
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int,
    hidden: TensorBox,
    weight: TensorBox,
    bias: TensorBox,
    cos_sin_cache: TensorBox,
    positions: TensorBox,
    cache_loc: TensorBox,
    k_cache: TensorBox,
    v_cache: TensorBox,
):
    q, k, v = fallback_gemm_rope(
        q_size,
        kv_size,
        head_dim,
        rotary_dim,
        hidden,
        weight,
        bias,
        cos_sin_cache,
        positions,
    )
    lowerings[aten.index_put_.default](k_cache, [cache_loc], k)
    lowerings[aten.index_put_.default](v_cache, [cache_loc], v)
    return (q, k, v)


def _allocate_rope_outputs(
    q_size: int,
    kv_size: int,
    hidden: TensorBox,
    *,
    layout=None,
):
    m_size = hidden.get_size()[0]
    device = hidden.get_device_or_error()
    hidden_dtype = hidden.get_dtype()
    total_n = q_size + 2 * kv_size
    layout = layout or FixedLayout(device, hidden_dtype, [m_size, q_size])
    k_buf = empty_strided([m_size, kv_size], None, dtype=hidden_dtype, device=device)
    # Match the original split view stride so inference can return V directly.
    v_buf = empty_strided(
        [m_size, kv_size],
        [total_n, 1],
        dtype=hidden_dtype,
        device=device,
    )
    return m_size, total_n, layout, k_buf, v_buf


def tuned_fused_gemm_rope(
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int,
    hidden: TensorBox,
    weight: TensorBox,
    bias: TensorBox,
    cos_sin_cache: TensorBox,
    positions: TensorBox,
    *,
    layout=None,
):
    if rotary_dim != head_dim or rotary_dim % 2 != 0:
        return fallback_gemm_rope(
            q_size,
            kv_size,
            head_dim,
            rotary_dim,
            hidden,
            weight,
            bias,
            cos_sin_cache,
            positions,
        )

    hidden.realize()
    weight.realize()
    bias.realize()
    cos_sin_cache.realize()
    positions.realize()

    m_size, total_n, layout, k_buf, v_buf = _allocate_rope_outputs(
        q_size,
        kv_size,
        hidden,
        layout=layout,
    )
    choices: list[TritonTemplateCaller] = []
    for config in gemm_rope_configs:
        gemm_rope_template.maybe_append_choice(
            choices,
            input_nodes=(hidden, weight, bias, cos_sin_cache, positions, k_buf, v_buf),
            layout=layout,
            mutated_inputs=[k_buf, v_buf],
            call_sizes=(m_size, total_n),
            BLOCK_N=head_dim,
            Q_SIZE=q_size,
            KV_SIZE=kv_size,
            HALF_ROTARY=rotary_dim // 2,
            GROUP_M=8,
            ALLOW_TF32=True,
            EVEN_K=hidden.get_size()[1] % config["BLOCK_K"] == 0,
            **config,
        )

    if not choices:
        return fallback_gemm_rope(
            q_size,
            kv_size,
            head_dim,
            rotary_dim,
            hidden,
            weight,
            bias,
            cos_sin_cache,
            positions,
        )

    q_out = autotune_select_algorithm(
        "gemm_rope",
        choices,
        [hidden, weight, bias, cos_sin_cache, positions, k_buf, v_buf],
        layout,
    )
    return (q_out, k_buf, v_buf)


def tuned_fused_gemm_rope_kv_cache(
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int,
    hidden: TensorBox,
    weight: TensorBox,
    bias: TensorBox,
    cos_sin_cache: TensorBox,
    positions: TensorBox,
    cache_loc: TensorBox,
    k_cache: TensorBox,
    v_cache: TensorBox,
    *,
    layout=None,
):
    if rotary_dim != head_dim or rotary_dim % 2 != 0:
        return fallback_gemm_rope_kv_cache(
            q_size,
            kv_size,
            head_dim,
            rotary_dim,
            hidden,
            weight,
            bias,
            cos_sin_cache,
            positions,
            cache_loc,
            k_cache,
            v_cache,
        )

    hidden.realize()
    weight.realize()
    bias.realize()
    cos_sin_cache.realize()
    positions.realize()
    cache_loc.realize()
    k_cache.realize()
    v_cache.realize()

    m_size, total_n, layout, k_buf, v_buf = _allocate_rope_outputs(
        q_size,
        kv_size,
        hidden,
        layout=layout,
    )
    choices: list[TritonTemplateCaller] = []
    for config in gemm_rope_configs:
        gemm_rope_kv_cache_template.maybe_append_choice(
            choices,
            input_nodes=(
                hidden,
                weight,
                bias,
                cos_sin_cache,
                positions,
                k_buf,
                v_buf,
                cache_loc,
                k_cache,
                v_cache,
            ),
            layout=layout,
            mutated_inputs=[k_buf, v_buf, k_cache, v_cache],
            call_sizes=(m_size, total_n),
            BLOCK_N=head_dim,
            Q_SIZE=q_size,
            KV_SIZE=kv_size,
            HALF_ROTARY=rotary_dim // 2,
            GROUP_M=8,
            ALLOW_TF32=True,
            EVEN_K=hidden.get_size()[1] % config["BLOCK_K"] == 0,
            **config,
        )

    if not choices:
        return fallback_gemm_rope_kv_cache(
            q_size,
            kv_size,
            head_dim,
            rotary_dim,
            hidden,
            weight,
            bias,
            cos_sin_cache,
            positions,
            cache_loc,
            k_cache,
            v_cache,
        )

    q_out = autotune_select_algorithm(
        "gemm_rope_kv_cache",
        choices,
        [
            hidden,
            weight,
            bias,
            cos_sin_cache,
            positions,
            k_buf,
            v_buf,
            cache_loc,
            k_cache,
            v_cache,
        ],
        layout,
    )
    return (q_out, k_buf, v_buf)


def gemm_rope_lowering_factory(
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int,
):
    return functools.partial(
        tuned_fused_gemm_rope,
        q_size,
        kv_size,
        head_dim,
        rotary_dim,
    )


def gemm_rope_kv_cache_lowering_factory(
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int,
):
    return functools.partial(
        tuned_fused_gemm_rope_kv_cache,
        q_size,
        kv_size,
        head_dim,
        rotary_dim,
    )


register_gemm_plugin(
    "rope_neox",
    lowering_factory=gemm_rope_lowering_factory,
)

register_gemm_plugin(
    "rope_neox_kv_cache",
    lowering_factory=gemm_rope_kv_cache_lowering_factory,
)
