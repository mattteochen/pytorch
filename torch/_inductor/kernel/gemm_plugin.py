# mypy: allow-untyped-defs
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class GemmPluginSpec:
    name: str
    lowering_factory: Callable[..., Callable[..., Any]]


_GEMM_PLUGIN_REGISTRY: dict[str, GemmPluginSpec] = {}


def register_gemm_plugin(
    name: str,
    *,
    lowering_factory: Callable[..., Callable[..., Any]],
) -> GemmPluginSpec:
    spec = GemmPluginSpec(name=name, lowering_factory=lowering_factory)
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
