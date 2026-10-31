"""Classical and plugin upscale algorithms (non-neural)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .classical import BUILTIN_ALGORITHMS, resize_classical

AlgorithmFn = Callable[[np.ndarray, int, int], np.ndarray]

_BUILTIN_META: dict[str, tuple[str, str]] = {
    "nearest": ("Nearest-neighbor", "Blocky, preserves hard edges — good for pixel art."),
    "bilinear": ("Bilinear", "Smooth linear interpolation."),
    "bicubic": ("Bicubic", "Smoother curves than bilinear; common photo default."),
    "lanczos2": ("Lanczos-2", "Sharper Lanczos (approx.)."),
    "lanczos3": ("Lanczos-3", "High-quality Lanczos — good general classical choice."),
}


@dataclass(frozen=True)
class AlgorithmInfo:
    name: str
    label: str
    group: str  # "builtin" | "paper" | "plugin"
    description: str = ""


_REGISTRY: dict[str, AlgorithmFn] = {}
_META: dict[str, AlgorithmInfo] = {}
_PLUGINS_LOADED = False


def _norm_algo(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def register_algorithm(
    name: str,
    fn: AlgorithmFn,
    *,
    overwrite: bool = False,
    label: str | None = None,
    group: str = "plugin",
    description: str = "",
) -> None:
    """Register a custom algorithm for later use (paper plugins, etc.)."""
    key = _norm_algo(name)
    if not key:
        raise ValueError("algorithm name required")
    if key in _REGISTRY and not overwrite:
        raise ValueError(f"algorithm already registered: {key}")
    _REGISTRY[key] = fn
    pretty = label or key.replace("_", " ").title()
    g = group if group in ("builtin", "paper", "plugin") else "plugin"
    if key in BUILTIN_ALGORITHMS:
        g = "builtin"
        bl, bd = _BUILTIN_META.get(key, (pretty, description))
        pretty = label or bl
        description = description or bd
    _META[key] = AlgorithmInfo(
        name=key, label=pretty, group=g, description=description or ""
    )


def unregister_algorithm(name: str) -> None:
    key = _norm_algo(name)
    _REGISTRY.pop(key, None)
    _META.pop(key, None)


def list_algorithms() -> list[str]:
    """Built-ins first (stable order), then any extra plugins."""
    ensure_plugins_loaded()
    builtins = list(BUILTIN_ALGORITHMS)
    extras = sorted(k for k in _REGISTRY if k not in BUILTIN_ALGORITHMS)
    return builtins + extras


def list_algorithm_meta() -> list[AlgorithmInfo]:
    """Metadata for UI / API: builtins then paper/plugin (sorted within group)."""
    ensure_plugins_loaded()
    out: list[AlgorithmInfo] = []
    for name in BUILTIN_ALGORITHMS:
        info = _META.get(name)
        if info:
            out.append(info)
        else:
            bl, bd = _BUILTIN_META.get(name, (name, ""))
            out.append(AlgorithmInfo(name=name, label=bl, group="builtin", description=bd))
    extras = [k for k in _REGISTRY if k not in BUILTIN_ALGORITHMS]
    extras.sort(key=lambda k: (_META[k].group if k in _META else "plugin", k))
    for k in extras:
        out.append(
            _META.get(k)
            or AlgorithmInfo(name=k, label=k.replace("_", " ").title(), group="plugin")
        )
    return out


def get_algorithm(name: str) -> AlgorithmFn | None:
    ensure_plugins_loaded()
    key = _norm_algo(name)
    if key in _REGISTRY:
        return _REGISTRY[key]
    if key in BUILTIN_ALGORITHMS:
        return lambda img, tw, th, _k=key: resize_classical(_k, img, tw, th)
    return None


def resize_with_algorithm(
    name: str, img: np.ndarray, target_w: int, target_h: int
) -> np.ndarray:
    """Resize HWC uint8 RGB (or single-channel) with a named algorithm."""
    fn = get_algorithm(name)
    if fn is None:
        raise ValueError(
            f"Unknown algorithm {name!r}. Available: {', '.join(list_algorithms())}"
        )
    tw, th = int(target_w), int(target_h)
    if tw < 1 or th < 1:
        raise ValueError("target size must be positive")
    return fn(img, tw, th)


def ensure_plugins_loaded() -> None:
    """Import plugin modules once so they can register_algorithm."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        from . import plugins as _plugins

        _plugins.load_plugins()
    except Exception:  # noqa: BLE001
        # Plugins are optional; classical built-ins still work.
        pass


# Register built-ins so plugins see a full catalog
for _name in BUILTIN_ALGORITHMS:
    _bl, _bd = _BUILTIN_META.get(_name, (_name, ""))
    register_algorithm(
        _name,
        lambda img, tw, th, _n=_name: resize_classical(_n, img, tw, th),
        overwrite=True,
        label=_bl,
        group="builtin",
        description=_bd,
    )

__all__ = [
    "AlgorithmFn",
    "AlgorithmInfo",
    "BUILTIN_ALGORITHMS",
    "ensure_plugins_loaded",
    "get_algorithm",
    "list_algorithm_meta",
    "list_algorithms",
    "register_algorithm",
    "resize_with_algorithm",
    "unregister_algorithm",
]
