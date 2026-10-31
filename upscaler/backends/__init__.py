"""Inference backends: torch (dev/full) and onnx (slim Windows desktop)."""

from __future__ import annotations

import os
from typing import Literal

BackendName = Literal["torch", "onnx", "auto"]


def resolve_backend(requested: str | None = None) -> Literal["torch", "onnx"]:
    """Pick backend. Env UPSCALE_BACKEND overrides; auto prefers onnx if importable and no torch."""
    raw = (requested or os.environ.get("UPSCALE_BACKEND") or "auto").strip().lower()
    if raw in ("torch", "pytorch"):
        return "torch"
    if raw in ("onnx", "ort"):
        return "onnx"
    # auto
    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        return "onnx"


__all__ = ["resolve_backend", "BackendName"]
