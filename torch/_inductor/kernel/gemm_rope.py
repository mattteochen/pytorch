# mypy: allow-untyped-defs
import functools

import torch

from .. import config as inductor_config
from ..ir import TensorBox
from ..kernel_inputs import MMKernelInputs
from ..lowering import lowerings
from ..template_heuristics.registry import get_template_heuristic
from .gemm_plugin import (
    get_gemm_plugin_shared_mm_extra_inputs,
    register_gemm_plugin,
)
from .mm import mm_template
from ..select_algorithm import autotune_select_algorithm, TritonTemplateCaller


aten = torch.ops.aten
GEMM_ROPE_PLUGIN = "rope_neox"
GEMM_ROPE_KV_CACHE_PLUGIN = "rope_neox_kv_cache"


shared_mm_rope_configs = [
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


def _render_shared_mm_rope_output(
    kernel,
    indices,
    val: str,
    *,
    mask: str | None = None,
    indent_width: int = 4,
    val_shape: tuple[str, ...] | None = None,
    block_indexing: bool = False,
    with_kv_cache: bool,
) -> str:
    assert tuple(indices) == ("idx_m", "idx_n")
    assert val == "acc"
    assert mask == "mask"
    assert indent_width == 4
    assert val_shape == ("BLOCK_M", "BLOCK_N")
    assert not block_indexing

    stride_out_m = kernel.stride(None, 0)
    stride_out_n = kernel.stride(None, 1)
    stride_cs = kernel.stride("COS_SIN", 0)
    output_ptr = kernel.output_ptr()

    lines = [
        "mask_m = rm < M",
        "mask_n = rn < N",
        f"stride_out_m = {stride_out_m}",
        f"stride_out_n = {stride_out_n}",
        f"stride_cs = {stride_cs}",
        "bias = tl.load(BIAS + rn, mask=mask_n, other=0.0).to(tl.float32)",
        "acc = acc + bias[None, :]",
        "col_start = pid_n * BLOCK_N",
        "if col_start < (Q_SIZE + KV_SIZE):",
        "    pos = tl.load(POS + rm, mask=mask_m, other=0)",
        "    cos = tl.load(",
        "        COS_SIN + pos[:, None] * stride_cs + tl.arange(0, HALF_ROTARY)[None, :],",
        "        mask=mask_m[:, None],",
        "        other=0.0,",
        "    )",
        "    sin = tl.load(",
        "        COS_SIN + pos[:, None] * stride_cs + HALF_ROTARY + tl.arange(0, HALF_ROTARY)[None, :],",
        "        mask=mask_m[:, None],",
        "        other=0.0,",
        "    )",
        "    acc_3d = tl.reshape(acc, (BLOCK_M, 2, HALF_ROTARY))",
        "    acc_t = tl.permute(acc_3d, (0, 2, 1))",
        "    x1, x2 = tl.split(acc_t)",
        "    o1 = x1 * cos - x2 * sin",
        "    o2 = x2 * cos + x1 * sin",
        "    out_joined = tl.join(o1, o2)",
        "    out_t = tl.permute(out_joined, (0, 2, 1))",
        "    acc = tl.reshape(out_t, (BLOCK_M, BLOCK_N))",
    ]

    if with_kv_cache:
        stride_ckm = kernel.stride("K_CACHE", 0)
        stride_ckn = kernel.stride("K_CACHE", 1)
        stride_cvm = kernel.stride("V_CACHE", 0)
        stride_cvn = kernel.stride("V_CACHE", 1)
        k_cache_rows = kernel.size("K_CACHE", 0)
        v_cache_rows = kernel.size("V_CACHE", 0)
        lines.extend(
            [
                f"stride_ckm = {stride_ckm}",
                f"stride_ckn = {stride_ckn}",
                f"stride_cvm = {stride_cvm}",
                f"stride_cvn = {stride_cvn}",
            ]
        )

    lines.extend(
        [
            f"# GEMM_ROPE_TRITON_ENTRANCE",
            f"tl.store({output_ptr} + idx_m * stride_out_m + idx_n * stride_out_n, acc, mask=mask)",
        ]
    )

    if with_kv_cache:
        lines.extend(
            [
                "if col_start >= Q_SIZE and col_start < (Q_SIZE + KV_SIZE):",
                f"    cache_rows = {k_cache_rows}",
                "    cache_row = tl.load(CACHE_LOC + rm, mask=mask_m, other=0)",
                "    cache_row = tl.where(cache_row < 0, cache_row + cache_rows, cache_row)",
                "    cache_mask = mask_m & (cache_row >= 0) & (cache_row < cache_rows)",
                "    cache_row = tl.where(cache_mask, cache_row, 0)",
                "    rk = rn - Q_SIZE",
                "    tl.store(",
                "        K_CACHE + cache_row[:, None] * stride_ckm + rk[None, :] * stride_ckn,",
                "        acc,",
                "        mask=cache_mask[:, None] & (rk[None, :] < KV_SIZE),",
                "    )",
                "elif col_start >= (Q_SIZE + KV_SIZE):",
                f"    cache_rows = {v_cache_rows}",
                "    cache_row = tl.load(CACHE_LOC + rm, mask=mask_m, other=0)",
                "    cache_row = tl.where(cache_row < 0, cache_row + cache_rows, cache_row)",
                "    cache_mask = mask_m & (cache_row >= 0) & (cache_row < cache_rows)",
                "    cache_row = tl.where(cache_mask, cache_row, 0)",
                "    rv = rn - Q_SIZE - KV_SIZE",
                "    tl.store(",
                "        V_CACHE + cache_row[:, None] * stride_cvm + rv[None, :] * stride_cvn,",
                "        acc,",
                "        mask=cache_mask[:, None] & (rv[None, :] < KV_SIZE),",
                "    )",
            ]
        )

    return "\n".join(
        [lines[0], *((" " * indent_width) + line if line else "" for line in lines[1:])]
    )


def render_shared_mm_rope_output(*args, **kwargs) -> str:
    return _render_shared_mm_rope_output(*args, **kwargs, with_kv_cache=False)


def render_shared_mm_rope_kv_cache_output(*args, **kwargs) -> str:
    return _render_shared_mm_rope_output(*args, **kwargs, with_kv_cache=True)


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
    weight_t: TensorBox,
    bias: TensorBox,
    cos_sin_cache: TensorBox,
    positions: TensorBox,
):
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
    weight_t: TensorBox,
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
        weight_t,
        bias,
        cos_sin_cache,
        positions,
    )
    lowerings[aten.index_put_.default](k_cache, [cache_loc], k)
    lowerings[aten.index_put_.default](v_cache, [cache_loc], v)
    return (q, k, v)


def _can_use_fused_rope(
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int,
) -> bool:
    return (
        rotary_dim == head_dim
        and rotary_dim % 2 == 0
        and q_size % head_dim == 0
        and kv_size % head_dim == 0
    )


def _build_shared_mm_rope_choices(
    plugin_name: str,
    kernel_inputs: MMKernelInputs,
    *,
    head_dim: int,
    q_size: int,
    kv_size: int,
    rotary_dim: int,
    layout,
    mutated_inputs=None,
) -> list[TritonTemplateCaller]:
    m_size, _, _ = kernel_inputs.mnk_symbolic()
    total_n = q_size + 2 * kv_size
    input_nodes = tuple(kernel_inputs.nodes())
    choices: list[TritonTemplateCaller] = []
    extra_input_names = get_gemm_plugin_shared_mm_extra_inputs(plugin_name)
    mm_heuristic = get_template_heuristic(
        mm_template.uid,
        kernel_inputs.device_type,
        "mm",
    )
    heuristic_extra_kwargs = mm_heuristic.get_extra_kwargs(kernel_inputs, "mm")
    config_kwargs_list: list[dict[str, object]] = []

    if inductor_config.max_autotune_gemm:
        for params in mm_heuristic.get_template_configs(kernel_inputs, "mm"):
            config_kwargs = {
                **heuristic_extra_kwargs,
                **params.to_kwargs(),
            }
            if config_kwargs.get("BLOCK_N") != head_dim:
                continue
            config_kwargs_list.append(config_kwargs)

    if not config_kwargs_list:
        config_kwargs_list.extend(
            {
                **heuristic_extra_kwargs,
                "BLOCK_N": head_dim,
                "GROUP_M": 8,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "ALLOW_TF32": heuristic_extra_kwargs["ALLOW_TF32"],
                "EVEN_K": input_nodes[0].get_size()[1] % config["BLOCK_K"] == 0,
                **config,
            }
            for config in shared_mm_rope_configs
        )

    for config_kwargs in config_kwargs_list:
        mm_template.maybe_append_choice(
            choices,
            input_nodes=input_nodes,
            layout=layout,
            mutated_inputs=mutated_inputs,
            call_sizes=(m_size, total_n),
            triton_meta={
                "MM_TEMPLATE_PLUGIN": plugin_name,
                "EXTRA_INPUT_NAMES": extra_input_names,
            },
            Q_SIZE=q_size,
            KV_SIZE=kv_size,
            HALF_ROTARY=rotary_dim // 2,
            **config_kwargs,
        )

    return choices


def tuned_fused_gemm_rope(
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int,
    hidden: TensorBox,
    weight_t: TensorBox,
    bias: TensorBox,
    cos_sin_cache: TensorBox,
    positions: TensorBox,
    *,
    layout=None,
):
    if not _can_use_fused_rope(q_size, kv_size, head_dim, rotary_dim):
        return fallback_gemm_rope(
            q_size,
            kv_size,
            head_dim,
            rotary_dim,
            hidden,
            weight_t,
            bias,
            cos_sin_cache,
            positions,
        )

    hidden.realize()
    weight_t.realize()
    bias.realize()
    cos_sin_cache.realize()
    positions.realize()

    kernel_inputs = MMKernelInputs(
        [hidden, weight_t, bias, cos_sin_cache, positions],
        mat1_idx=0,
        mat2_idx=1,
    )
    layout = layout or kernel_inputs.output_layout(flexible=False)
    choices = _build_shared_mm_rope_choices(
        GEMM_ROPE_PLUGIN,
        kernel_inputs,
        head_dim=head_dim,
        q_size=q_size,
        kv_size=kv_size,
        rotary_dim=rotary_dim,
        layout=layout,
    )

    if not choices:
        return fallback_gemm_rope(
            q_size,
            kv_size,
            head_dim,
            rotary_dim,
            hidden,
            weight_t,
            bias,
            cos_sin_cache,
            positions,
        )

    qkv_out = autotune_select_algorithm(
        "gemm_rope", choices, kernel_inputs.nodes(), layout
    )
    q, k, v = lowerings[aten.split_with_sizes.default](
        qkv_out, [q_size, kv_size, kv_size], -1
    )
    return (q, k, v)


def tuned_fused_gemm_rope_kv_cache(
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int,
    hidden: TensorBox,
    weight_t: TensorBox,
    bias: TensorBox,
    cos_sin_cache: TensorBox,
    positions: TensorBox,
    cache_loc: TensorBox,
    k_cache: TensorBox,
    v_cache: TensorBox,
    *,
    layout=None,
):
    if not _can_use_fused_rope(q_size, kv_size, head_dim, rotary_dim):
        return fallback_gemm_rope_kv_cache(
            q_size,
            kv_size,
            head_dim,
            rotary_dim,
            hidden,
            weight_t,
            bias,
            cos_sin_cache,
            positions,
            cache_loc,
            k_cache,
            v_cache,
        )

    hidden.realize()
    weight_t.realize()
    bias.realize()
    cos_sin_cache.realize()
    positions.realize()
    cache_loc.realize()
    k_cache.realize()
    v_cache.realize()

    kernel_inputs = MMKernelInputs(
        [hidden, weight_t, bias, cos_sin_cache, positions, cache_loc, k_cache, v_cache],
        mat1_idx=0,
        mat2_idx=1,
    )
    layout = layout or kernel_inputs.output_layout(flexible=False)
    choices = _build_shared_mm_rope_choices(
        GEMM_ROPE_KV_CACHE_PLUGIN,
        kernel_inputs,
        head_dim=head_dim,
        q_size=q_size,
        kv_size=kv_size,
        rotary_dim=rotary_dim,
        layout=layout,
        mutated_inputs=[k_cache, v_cache],
    )

    if not choices:
        return fallback_gemm_rope_kv_cache(
            q_size,
            kv_size,
            head_dim,
            rotary_dim,
            hidden,
            weight_t,
            bias,
            cos_sin_cache,
            positions,
            cache_loc,
            k_cache,
            v_cache,
        )

    qkv_out = autotune_select_algorithm(
        "gemm_rope_kv_cache", choices, kernel_inputs.nodes(), layout
    )
    q, k, v = lowerings[aten.split_with_sizes.default](
        qkv_out, [q_size, kv_size, kv_size], -1
    )
    return (q, k, v)


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
    GEMM_ROPE_PLUGIN,
    lowering_factory=gemm_rope_lowering_factory,
    shared_mm_extra_inputs=("BIAS", "COS_SIN", "POS"),
    shared_mm_finalizer=render_shared_mm_rope_output,
)

register_gemm_plugin(
    GEMM_ROPE_KV_CACHE_PLUGIN,
    lowering_factory=gemm_rope_kv_cache_lowering_factory,
    shared_mm_extra_inputs=("BIAS", "COS_SIN", "POS", "CACHE_LOC", "K_CACHE", "V_CACHE"),
    shared_mm_finalizer=render_shared_mm_rope_kv_cache_output,
)
