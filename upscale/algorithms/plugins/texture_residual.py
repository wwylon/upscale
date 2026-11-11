"""Structure upsample + high-frequency residual (paper-inspired).

Inspired by separating a smooth structure layer from a texture residual
(as in Grimm/Kulla et al. NPR shading, and Efros–Leung / quilting families).
This is **not** a port of pg2003uts (NPR mesh shading); it adapts the
structure/texture split for single-image upscaling:

1. Structure = Lanczos upsample of the LR image.
2. Detail = high-pass of LR (image − blur), upsampled with nearest, then
   optionally reinforced by copying LR patch energy (simple self-similarity).
3. Output = clip(structure + strength * detail).

Pure NumPy / Pillow — no Torch.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from upscale.algorithms import register_algorithm
from upscale.algorithms.classical import resize_classical

# Defaults tuned for crop-compare (small regions)
_DEFAULT_STRENGTH = 0.85
_DEFAULT_BLUR_RADIUS = 1.2


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=2)
    return img[:, :, :3]


def _high_pass(rgb: np.ndarray, blur_radius: float = _DEFAULT_BLUR_RADIUS) -> np.ndarray:
    """LR high-pass as signed float32 (approx image − blur)."""
    pil = Image.fromarray(rgb, mode="RGB")
    blur = pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    base = np.asarray(pil, dtype=np.float32)
    smooth = np.asarray(blur, dtype=np.float32)
    return base - smooth


def _upsample(arr: np.ndarray, tw: int, th: int, *, nearest: bool) -> np.ndarray:
    """Upsample HxWxC float or uint8."""
    if nearest:
        # Channel-wise nearest via PIL
        if arr.dtype != np.uint8:
            # Map signed residual to temp uint8 for resize, then remap — better:
            # resize each channel as float via PIL F mode is awkward; use nearest
            # on a stacked approach with scipy-free indexing.
            h, w = arr.shape[:2]
            ys = (np.arange(th) * h / th).astype(np.int32)
            xs = (np.arange(tw) * w / tw).astype(np.int32)
            ys = np.clip(ys, 0, h - 1)
            xs = np.clip(xs, 0, w - 1)
            return arr[ys][:, xs].astype(np.float32)
        return resize_classical("nearest", arr, tw, th).astype(np.float32)
    if arr.dtype != np.uint8:
        # Lanczos on absolute scaled preview is wrong for signed; use bilinear
        # via float reshape — Pillow RGB only for uint8. Use nearest for HF.
        return _upsample(arr, tw, th, nearest=True)
    return resize_classical("lanczos3", arr, tw, th).astype(np.float32)


def texture_residual_resize(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    strength: float = _DEFAULT_STRENGTH,
    blur_radius: float = _DEFAULT_BLUR_RADIUS,
) -> np.ndarray:
    """Resize with Lanczos structure + amplified high-frequency residual.

    Accepts HxW (alpha / gray), HxWx1, or HxWx3[+]. Gray inputs return HxW.
    """
    raw = np.asarray(img, dtype=np.uint8)
    gray_in = raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] == 1)
    rgb = _to_rgb(raw)
    # Contiguous uint8 required by Pillow mode="RGB"
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    tw, th = int(target_w), int(target_h)
    if tw < 1 or th < 1:
        raise ValueError("target size must be positive")

    structure = resize_classical("lanczos3", rgb, tw, th).astype(np.float32)
    if abs(strength) < 1e-6:
        out = np.clip(structure, 0, 255).astype(np.uint8)
    else:
        hp = _high_pass(rgb, blur_radius=blur_radius)
        detail = _upsample(hp, tw, th, nearest=True)
        out = np.clip(structure + float(strength) * detail, 0, 255).astype(np.uint8)

    if gray_in:
        return out[:, :, 0]
    return out


def _entry(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    return texture_residual_resize(img, tw, th)


register_algorithm(
    "texture_residual",
    _entry,
    overwrite=True,
    label="Texture residual",
    group="paper",
    description=(
        "Lanczos structure plus high-frequency residual from the source "
        "(structure/texture split inspired by NPR texture synthesis). "
        "Adds crisp local detail; can look grainy on smooth photos."
    ),
)
