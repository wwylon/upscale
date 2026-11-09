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


def _high_pass_noise(gray: np.ndarray, radius: int = 3) -> float:
    """Median absolute residual after a box blur — true high-frequency noise."""
    from PIL import Image, ImageFilter

    g8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    soft = np.asarray(
        Image.fromarray(g8, mode="L").filter(ImageFilter.BoxBlur(radius)),
        dtype=np.float32,
    ) / 255.0
    residual = gray - soft
    return float(np.median(np.abs(residual)))


def analyze_image(path: Path | str) -> dict[str, float]:
    """Return simple stats used for classification."""
    arr = _load_rgb(Path(path))
    gray = arr.mean(axis=2)
    gy, gx = np.gradient(gray)
    lap_var = float(np.var(gx) + np.var(gy))
    color_std = float(arr.std())
    noise = _high_pass_noise(gray)
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
    # High-pass residual threshold (tuned for 0–1 float images).
    if stats["noise_median"] > 0.025:
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
    denoise: float | None = None
    if fast:
        model = "4x-spanx4-ch48"
        reason = "Fast SPAN preset requested."
    elif kind == "noisy":
        model = "realesr-general-wdn-x4v3"
        # Away from the 0.5 "leave base alone" sentinel — favor WDN.
        denoise = 0.75
        reason = "Elevated high-frequency noise — WDN weights with denoise blend."
    elif kind == "art":
        model = "RealESRGAN_x4plus_anime_6B"
        reason = "Strong edges / flat colors — anime-tuned model."
    elif kind == "flat":
        model = "4x-UltraSharp"
        reason = "Low detail — sharpening-oriented model."
    else:
        model = "RealESRGAN_x4plus"
        reason = "Balanced photo content — default Real-ESRGAN."
    out: dict[str, Any] = {
        "kind": kind,
        "model": model,
        "reason": reason,
        "stats": stats,
    }
    if denoise is not None:
        out["denoise"] = denoise
    return out
