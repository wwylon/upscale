"""Pre/post image filters for the upscale pipeline."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .schema import FilterSettings


def apply_pre_filters(img: np.ndarray, settings: FilterSettings) -> np.ndarray:
    """Denoise / JPEG clean / deblur before the model."""
    out = img
    if settings.jpeg_clean > 0:
        out = _jpeg_clean(out, settings.jpeg_clean)
    if settings.pre_denoise > 0:
        out = _denoise(out, settings.pre_denoise)
    if settings.deblur > 0:
        out = _deblur(out, settings.deblur)
    return out


def apply_post_filters(img: np.ndarray, settings: FilterSettings) -> np.ndarray:
    """Edge enhance / grain / color / halo reduce after the model."""
    out = img
    if settings.edge > 0:
        out = _edge_enhance(out, settings.edge)
    if settings.halo_reduce > 0:
        out = _halo_reduce(out, settings.halo_reduce)
    if settings.color_contrast > 0:
        out = _color_contrast(out, settings.color_contrast)
    if settings.grain > 0:
        out = _film_grain(out, settings.grain)
    # face_restore intentionally unused until a restorer is registered
    return out


def estimate_time_factor(settings: FilterSettings) -> float:
    """Rough multiplier for time estimates when filters are on."""
    f = 1.0
    if settings.pre_denoise > 0:
        f += 0.05 * settings.pre_denoise
    if settings.jpeg_clean > 0:
        f += 0.08 * settings.jpeg_clean
    if settings.deblur > 0:
        f += 0.06 * settings.deblur
    if settings.edge > 0:
        f += 0.03 * settings.edge
    if settings.grain > 0:
        f += 0.02 * settings.grain
    if settings.color_contrast > 0:
        f += 0.02 * settings.color_contrast
    if settings.halo_reduce > 0:
        f += 0.04 * settings.halo_reduce
    # face_restore is a stub — do not inflate ETA until a restorer is wired.
    return f


def any_pre_filters(settings: FilterSettings) -> bool:
    return (
        settings.pre_denoise > 0
        or settings.jpeg_clean > 0
        or settings.deblur > 0
        or settings.seamless
    )


def any_post_filters(settings: FilterSettings) -> bool:
    return (
        settings.edge > 0
        or settings.grain > 0
        or settings.color_contrast > 0
        or settings.halo_reduce > 0
        or settings.face_restore > 0
    )


def _denoise(img: np.ndarray, amount: float) -> np.ndarray:
    radius = max(0.5, 2.0 * amount)
    pil = Image.fromarray(img)
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    a = float(np.clip(amount, 0, 1))
    mix = Image.blend(pil, blurred, a * 0.6)
    return np.array(mix, dtype=np.uint8)


def _jpeg_clean(img: np.ndarray, amount: float) -> np.ndarray:
    # Mild bilateral-like: median then blend
    pil = Image.fromarray(img)
    cleaned = pil.filter(ImageFilter.MedianFilter(size=3 if amount < 0.7 else 5))
    a = float(np.clip(amount, 0, 1))
    return np.array(Image.blend(pil, cleaned, a * 0.7), dtype=np.uint8)


def _deblur(img: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp-mask style sharpening before SR (helps soft / motion-soft sources)."""
    a = float(np.clip(amount, 0, 1))
    radius = 1.0 + 1.5 * a
    percent = int(80 + 120 * a)
    pil = Image.fromarray(img)
    sharp = pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=2))
    return np.array(Image.blend(pil, sharp, a * 0.85), dtype=np.uint8)


def _edge_enhance(img: np.ndarray, amount: float) -> np.ndarray:
    pil = Image.fromarray(img)
    sharp = pil.filter(ImageFilter.EDGE_ENHANCE_MORE if amount > 0.5 else ImageFilter.EDGE_ENHANCE)
    a = float(np.clip(amount, 0, 1))
    return np.array(Image.blend(pil, sharp, a), dtype=np.uint8)


def _film_grain(img: np.ndarray, amount: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 12.0 * amount, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255)
    return out.astype(np.uint8)


def _color_contrast(img: np.ndarray, amount: float) -> np.ndarray:
    """Mild color + contrast boost after SR (0 = identity, 1 = strong)."""
    a = float(np.clip(amount, 0, 1))
    pil = Image.fromarray(img)
    # Enhance factors: 1.0 = unchanged; map amount → up to ~1.35
    factor = 1.0 + 0.35 * a
    pil = ImageEnhance.Contrast(pil).enhance(factor)
    pil = ImageEnhance.Color(pil).enhance(1.0 + 0.25 * a)
    return np.array(pil, dtype=np.uint8)


def _halo_reduce(img: np.ndarray, amount: float) -> np.ndarray:
    """Pull bright/dark overshoots toward a median neighborhood (reduces ringing)."""
    a = float(np.clip(amount, 0, 1))
    pil = Image.fromarray(img)
    soft = pil.filter(ImageFilter.MedianFilter(size=3))
    soft_a = np.array(soft, dtype=np.float32)
    src = img.astype(np.float32)
    # Where |src - median| is large, blend toward median
    delta = np.abs(src - soft_a)
    strength = np.clip(delta / 40.0, 0.0, 1.0) * a * 0.75
    out = src * (1.0 - strength) + soft_a * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def wrap_seamless(img: np.ndarray, pad: int) -> np.ndarray:
    """Pad image with wraparound for seamless texture mode."""
    if pad <= 0:
        return img
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="wrap")


def unwrap_seamless(img: np.ndarray, pad: int, scale: int) -> np.ndarray:
    if pad <= 0:
        return img
    p = pad * scale
    return img[p:-p, p:-p]
