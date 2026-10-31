"""Heuristic model suggestion from image statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _load_rgb(path: Path, max_side: int = 512) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        return np.asarray(im, dtype=np.float32) / 255.0


def analyze_image(path: Path | str) -> dict[str, float]:
    """Return simple stats used for classification."""
    arr = _load_rgb(Path(path))
    gray = arr.mean(axis=2)
    # Laplacian variance — sharpness / edge density
    gy, gx = np.gradient(gray)
    lap_var = float(np.var(gx) + np.var(gy))
    color_std = float(arr.std())
    # High-frequency noise proxy
    noise = float(np.median(np.abs(gray - np.median(gray))))
    flat = float(1.0 - np.std(gray))
    return {
        "laplacian_var": lap_var,
        "color_std": color_std,
        "noise_median": noise,
        "flatness": flat,
    }


def classify_kind(stats: dict[str, float]) -> str:
    if stats["flatness"] > 0.92 and stats["laplacian_var"] < 0.002:
        return "flat"
    if stats["noise_median"] > 0.08:
        return "noisy"
    if stats["laplacian_var"] > 0.015 and stats["color_std"] < 0.22:
        return "art"
    return "photo"


def suggest_model(path: Path | str, *, fast: bool = False) -> dict[str, Any]:
    """
    Suggest a model name and rationale from image heuristics.
    """
    stats = analyze_image(path)
    kind = classify_kind(stats)
    if fast:
        model = "4x-spanx4-ch48"
        reason = "Fast SPAN preset requested."
    elif kind == "noisy":
        model = "realesr-general-x4v3"
        reason = "Elevated noise — general denoise-friendly model."
    elif kind == "art":
        model = "RealESRGAN_x4plus_anime_6B"
        reason = "Strong edges / flat colors — anime-tuned model."
    elif kind == "flat":
        model = "4x-UltraSharp"
        reason = "Low detail — sharpening-oriented model."
    else:
        model = "RealESRGAN_x4plus"
        reason = "Balanced photo content — default Real-ESRGAN."
    return {
        "kind": kind,
        "model": model,
        "reason": reason,
        "stats": stats,
    }
