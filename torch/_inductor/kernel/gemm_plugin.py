# mypy: allow-untyped-defs
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class GemmPluginSpec:
    name: str
    lowering_factory: Callable[..., Callable[..., Any]]
    shared_mm_extra_inputs: tuple[str, ...] = ()
    shared_mm_finalizer: Optional[Callable[..., str]] = None


_GEMM_PLUGIN_REGISTRY: dict[str, GemmPluginSpec] = {}


def register_gemm_plugin(
    name: str,
    *,
    lowering_factory: Callable[..., Callable[..., Any]],
    shared_mm_extra_inputs: tuple[str, ...] = (),
    shared_mm_finalizer: Optional[Callable[..., str]] = None,
) -> GemmPluginSpec:
    spec = GemmPluginSpec(
        name=name,
        lowering_factory=lowering_factory,
        shared_mm_extra_inputs=shared_mm_extra_inputs,
        shared_mm_finalizer=shared_mm_finalizer,
    )
    existing = _GEMM_PLUGIN_REGISTRY.get(name)
    if existing is not None and existing != spec:
        raise AssertionError(f"duplicate GEMM plugin registration: {name}")
    _GEMM_PLUGIN_REGISTRY[name] = spec
    return spec


def get_gemm_plugin(name: str) -> GemmPluginSpec:
    return _GEMM_PLUGIN_REGISTRY[name]


def has_gemm_plugin(name: str) -> bool:
    return name in _GEMM_PLUGIN_REGISTRY


def make_gemm_plugin_lowering(name: str, /, *plugin_args: Any) -> Callable[..., Any]:
    spec = get_gemm_plugin(name)
    lowering = spec.lowering_factory(*plugin_args)
    lowering.__name__ = f"tuned_gemm_plugin_{name}"
    lowering._inductor_lowering_function = True  # type: ignore[attr-defined]
    return lowering


def get_gemm_plugin_shared_mm_extra_inputs(name: str) -> tuple[str, ...]:
    return get_gemm_plugin(name).shared_mm_extra_inputs


def render_gemm_plugin_shared_mm_output(
    name: str,
    kernel: Any,
    indices,
    val: str,
    *,
    mask: Optional[str] = None,
    indent_width: int = 4,
    val_shape: Optional[tuple[str, ...]] = None,
    block_indexing: bool = False,
) -> str:
    spec = get_gemm_plugin(name)
    if spec.shared_mm_finalizer is None:
        raise AssertionError(f"GEMM plugin {name} does not define a shared-mm finalizer")
    return spec.shared_mm_finalizer(
        kernel,
        indices,
        val,
        mask=mask,
        indent_width=indent_width,
        val_shape=val_shape,
        block_indexing=block_indexing,
    )
