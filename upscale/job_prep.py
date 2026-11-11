"""Shared prepare / finalize for classical, Torch, and ONNX execution paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import click
import numpy as np

from .config import UpscaleConfig
from .filters import (
    any_post_filters,
    any_pre_filters,
    apply_post_filters,
    apply_pre_filters,
    unwrap_seamless,
    wrap_seamless,
)
from .io_utils import lanczos_resize, load_rgba_with_meta, save_image
from .naming import resolve_output_path
from .schema import FilterSettings

if TYPE_CHECKING:
    from .io_utils import ImageMeta


@dataclass
class PreparedJob:
    cfg: UpscaleConfig
    img: np.ndarray
    """Image after crop + pre-filters (+ optional seamless pad)."""
    alpha: np.ndarray | None
    original_mode: str
    image_meta: ImageMeta | None
    content_w: int
    content_h: int
    """Unpadded content size (for targets and unwrap)."""
    unpadded: np.ndarray
    """Pre-filter image before seamless pad (for strength blend)."""
    scale: float
    """Cover/scale factor used for SR."""
    target_w: int
    target_h: int
    """Exact output dimensions (W×H exact when both set)."""
    pad: int
    filters: FilterSettings
    show: bool
    log: Callable[[str], None]


# Soft cap on output pixels for a single job (≈ 200 megapixels).
MAX_OUTPUT_PIXELS = 200_000_000


def validate_sizing(cfg: UpscaleConfig) -> None:
    if cfg.scale is not None and cfg.target_size is not None:
        raise click.UsageError("Specify --scale or --target-size, not both.")
    has_scale = cfg.scale is not None
    has_target = cfg.target_size is not None
    has_wh = cfg.width is not None and cfg.height is not None
    if (cfg.width is None) ^ (cfg.height is None):
        raise click.UsageError("Specify both width and height, or neither.")
    if not (has_scale or has_target or has_wh):
        raise click.UsageError("Specify --scale, --target-size, or width×height.")


def resolve_scale_and_targets(
    cfg: UpscaleConfig, content_w: int, content_h: int
) -> tuple[float, int, int]:
    """Return (scale_for_sr, exact_target_w, exact_target_h)."""
    scale = cfg.scale
    if cfg.width is not None and cfg.height is not None:
        scale = max(cfg.width / content_w, cfg.height / content_h)
        tw, th = int(cfg.width), int(cfg.height)
    elif cfg.target_size is not None:
        scale = cfg.target_size / max(content_h, content_w)
        tw = round(content_w * scale)
        th = round(content_h * scale)
    else:
        assert scale is not None
        tw = round(content_w * scale)
        th = round(content_h * scale)
    if scale is None or scale <= 0:
        raise click.UsageError("Scale must be positive.")
    if tw <= 0 or th <= 0:
        raise click.UsageError("Output dimensions must be positive.")
    if tw * th > MAX_OUTPUT_PIXELS:
        raise click.UsageError(
            f"Output {tw}×{th} exceeds the {MAX_OUTPUT_PIXELS:,} pixel budget."
        )
    return float(scale), tw, th


def clamp_overlap(overlap: int, tile: int) -> int:
    overlap = max(0, int(overlap))
    if overlap >= tile // 2:
        return max(0, tile // 4)
    return overlap


def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """Rounded (w, h) after multiplying by scale."""
    return round(width * scale), round(height * scale)


def prepare_job(
    cfg: UpscaleConfig,
    *,
    on_progress: Callable[[dict], None] | None = None,
    echo: Callable[[str], None] | None = None,
) -> PreparedJob:
    if cfg.input is None or cfg.output is None:
        raise ValueError("input and output paths are required")
    validate_sizing(cfg)

    show = not cfg.quiet
    log = echo if echo is not None else (click.echo if show else (lambda _m: None))

    if show:
        log(f"Loading {cfg.input} ...")
    img, alpha, original_mode, image_meta = load_rgba_with_meta(cfg.input)
    cfg.image_meta = image_meta
    H, W = img.shape[:2]

    if cfg.crop is not None:
        x, y, cw, ch = cfg.crop
        x = max(0, int(x))
        y = max(0, int(y))
        x2 = min(W, x + max(0, int(cw)))
        y2 = min(H, y + max(0, int(ch)))
        if x2 <= x or y2 <= y:
            raise click.UsageError("Crop rectangle is empty or outside the image.")
        img = img[y:y2, x:x2].copy()
        if alpha is not None:
            alpha = alpha[y:y2, x:x2].copy()
        H, W = img.shape[:2]

    scale, target_w, target_h = resolve_scale_and_targets(cfg, W, H)
    if scale < 1.0 and show:
        log(f"Scale {scale:.3f}x is a downscale — applying Lanczos only.")

    filters = cfg.filters if cfg.filters is not None else FilterSettings.model_validate({})
    if on_progress and any_pre_filters(filters):
        on_progress({"type": "pre_filters", "pct": 0.01, "message": "Applying filters…"})
    img = apply_pre_filters(img, filters)
    unpadded = img
    content_h, content_w = img.shape[:2]
    # Recompute targets from content after filters (same size).
    scale, target_w, target_h = resolve_scale_and_targets(cfg, content_w, content_h)

    pad = 16 if filters.seamless else 0
    if pad:
        img = wrap_seamless(img, pad)
        # Keep alpha geometry aligned with RGB when SR-alpha is used.
        if alpha is not None:
            alpha = wrap_seamless(alpha, pad)

    return PreparedJob(
        cfg=cfg,
        img=img,
        alpha=alpha,
        original_mode=original_mode,
        image_meta=image_meta,
        content_w=content_w,
        content_h=content_h,
        unpadded=unpadded,
        scale=scale,
        target_w=target_w,
        target_h=target_h,
        pad=pad,
        filters=filters,
        show=show,
        log=log,
    )


def seamless_unwrap_margins(
    result: np.ndarray, content_w: int, content_h: int, pad: int
) -> tuple[int, int]:
    """Output-space pad margins (pad_y, pad_x) for non-integer scales too."""
    if pad <= 0:
        return 0, 0
    padded_w = content_w + 2 * pad
    padded_h = content_h + 2 * pad
    py = int(round(pad * result.shape[0] / max(padded_h, 1)))
    px = int(round(pad * result.shape[1] / max(padded_w, 1)))
    return max(0, py), max(0, px)


def seamless_unwrap_scale(
    result: np.ndarray, content_w: int, content_h: int, pad: int
) -> int:
    """Backward-compatible integer factor (max of axis ratios, rounded).

    Prefer :func:`seamless_unwrap_margins` for correct non-integer crops.
    """
    py, px = seamless_unwrap_margins(result, content_w, content_h, pad)
    if pad <= 0:
        return 1
    # Approximate factor for callers still using unwrap_seamless(..., scale=).
    return max(1, int(round(max(py, px) / max(pad, 1))))


def apply_strength(
    result: np.ndarray,
    unpadded: np.ndarray,
    target_w: int,
    target_h: int,
    strength: float,
) -> np.ndarray:
    if strength is None or strength >= 1.0:
        return result
    a = float(np.clip(strength, 0.0, 1.0))
    plain = lanczos_resize(unpadded, target_w, target_h)
    return np.clip(
        result.astype(np.float32) * a + plain.astype(np.float32) * (1.0 - a),
        0,
        255,
    ).astype(np.uint8)


def finalize_result(
    prep: PreparedJob,
    result: np.ndarray,
    *,
    on_progress: Callable[[dict], None] | None = None,
    model_label: str | None = None,
    out_alpha: np.ndarray | None = None,
) -> Path:
    """Unwrap seamless → strength → post-filters → exact size → save."""
    cfg = prep.cfg
    pad = prep.pad
    margins: tuple[int, int] | None = None
    if pad:
        margins = seamless_unwrap_margins(
            result, prep.content_w, prep.content_h, pad
        )
        result = unwrap_seamless(result, pad, margins=margins)

    # Exact W×H (or scale-derived targets)
    if result.shape[1] != prep.target_w or result.shape[0] != prep.target_h:
        result = lanczos_resize(result, prep.target_w, prep.target_h)

    result = apply_strength(
        result, prep.unpadded, prep.target_w, prep.target_h, cfg.strength
    )

    if on_progress and any_post_filters(prep.filters):
        on_progress({"type": "filters", "pct": 0.97, "message": "Applying filters…"})
    result = apply_post_filters(result, prep.filters)

    if out_alpha is None:
        out_alpha = prep.alpha
    if out_alpha is not None and prep.original_mode in ("RGBA", "LA"):
        a = np.asarray(out_alpha)
        if a.ndim == 3:
            a = a[:, :, 0]
        a = np.ascontiguousarray(a, dtype=np.uint8)
        if pad:
            padded_h = prep.content_h + 2 * pad
            padded_w = prep.content_w + 2 * pad
            if a.shape[0] == padded_h and a.shape[1] == padded_w:
                # Still at padded input resolution (classical / Lanczos alpha).
                a = unwrap_seamless(a, pad, margins=(pad, pad))
            elif margins is not None:
                # SR'd to padded output size — use the same margins as RGB.
                a = unwrap_seamless(a, pad, margins=margins)
        if a.shape[0] != result.shape[0] or a.shape[1] != result.shape[1]:
            from PIL import Image

            a = np.array(
                Image.fromarray(a, mode="L").resize(
                    (result.shape[1], result.shape[0]), Image.Resampling.LANCZOS
                ),
                dtype=np.uint8,
            )
        out_alpha = a

    if on_progress:
        on_progress({"type": "writing", "pct": 0.99, "message": "Writing output…"})

    out_path = Path(cfg.output)  # type: ignore[arg-type]
    label = model_label or cfg.model
    if cfg.name_pattern is not None:
        resolved = resolve_output_path(
            out_path.parent if out_path.suffix else out_path,
            stem=Path(cfg.input).stem,  # type: ignore[arg-type]
            scale=prep.scale,
            model=label,
            ext=(
                cfg.format
                if cfg.format != "auto"
                else (out_path.suffix.lstrip(".") or "png")
            ),
            pattern=cfg.name_pattern,
            width=result.shape[1],
            height=result.shape[0],
        )
        if resolved is None:
            raise click.UsageError("Output exists and on-exists=skip")
        out_path = resolved

    actual_path = save_image(
        result,
        out_path,
        prep.original_mode,
        out_alpha,
        cfg.tiff_compression,
        fmt=cfg.format,
        quality=cfg.quality,
        meta=prep.image_meta,
    )
    if prep.show:
        out_h, out_w = result.shape[:2]
        prep.log(f"Saved {out_w}×{out_h} → {actual_path}")
    return actual_path
