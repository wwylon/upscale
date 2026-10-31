"""Crop compare: same region through models, classical algos, or filters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import UpscaleConfig
from .engine import execute
from .schema import CropRect, FilterSettings


# UI filter key → FilterSettings field
FILTER_UI_TO_SCHEMA = {
    "denoise": "pre_denoise",
    "jpeg": "jpeg_clean",
    "deblur": "deblur",
    "edge": "edge",
    "grain": "grain",
    "color": "color_contrast",
    "halo": "halo_reduce",
}


def _safe_id(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _filters_for_key(
    key: str | None, amounts: dict[str, float] | None
) -> FilterSettings:
    """Build FilterSettings with at most one UI filter enabled (or all off)."""
    amounts = amounts or {}
    kwargs: dict[str, Any] = {v: 0.0 for v in FILTER_UI_TO_SCHEMA.values()}
    kwargs["seamless"] = False
    kwargs["face_restore"] = 0.0
    if key and key != "none" and key in FILTER_UI_TO_SCHEMA:
        field = FILTER_UI_TO_SCHEMA[key]
        amt = float(amounts.get(key, 0.35))
        kwargs[field] = max(0.0, min(1.0, amt))
    return FilterSettings(**kwargs)


def build_variants(
    *,
    models: list[str] | None = None,
    algorithms: list[str] | None = None,
    filters: list[str] | None = None,
    filter_amounts: dict[str, float] | None = None,
    base_model: str = "RealESRGAN_x4plus",
    base_algorithm: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize UI selections into executable compare variants."""
    variants: list[dict[str, Any]] = []
    for m in models or []:
        if not m:
            continue
        variants.append(
            {
                "id": m,
                "label": m,
                "kind": "model",
                "model": m,
                "algorithm": None,
                "filters": FilterSettings(),
            }
        )
    for a in algorithms or []:
        if not a:
            continue
        variants.append(
            {
                "id": f"algo:{a}",
                "label": f"Classical · {a}",
                "kind": "algorithm",
                "model": base_model,
                "algorithm": a,
                "filters": FilterSettings(),
            }
        )
    for fk in filters or []:
        if not fk:
            continue
        label = "No filters" if fk == "none" else f"Filter · {fk}"
        variants.append(
            {
                "id": f"filt:{fk}",
                "label": label,
                "kind": "filter",
                "model": base_model,
                "algorithm": base_algorithm,
                "filters": _filters_for_key(fk, filter_amounts),
            }
        )
    return variants


def compare_variants(
    image: Path,
    out_dir: Path,
    *,
    variants: list[dict[str, Any]],
    crop: CropRect | tuple[int, int, int, int],
    scale: float = 4.0,
    device: str = "auto",
    half: bool = False,
    progress: Callable[[str, float], None] | None = None,
    cancel_event: Any | None = None,
) -> list[dict]:
    """
    Upscale the same crop for each variant. Returns list of
    {id, model, label, path, seconds, error, kind}.
    ``model`` mirrors ``id`` for API path compatibility.

    When ``cancel_event`` is set, stops between variants (and mid-variant via
    ``execute``) and returns whatever finished so far.
    """
    import time

    from .tiling import JobCancelled

    if isinstance(crop, tuple):
        crop = CropRect(x=crop[0], y=crop[1], w=crop[2], h=crop[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    n = max(len(variants), 1)
    for i, var in enumerate(variants):
        if cancel_event is not None and cancel_event.is_set():
            break
        vid = str(var.get("id") or var.get("model") or f"v{i}")
        label = str(var.get("label") or vid)
        kind = str(var.get("kind") or "model")
        if progress:
            progress(f"compare {label}", i / n)
        out = out_dir / f"compare_{_safe_id(vid)}.png"
        t0 = time.perf_counter()
        err = None
        path = None
        try:
            algorithm = var.get("algorithm")
            filt = var.get("filters")
            if not isinstance(filt, FilterSettings):
                filt = FilterSettings.model_validate(filt or {})
            cfg = UpscaleConfig(
                input=image,
                output=out,
                scale=scale,
                model=str(var.get("model") or "RealESRGAN_x4plus"),
                algorithm=algorithm,
                device=device,
                half=half,
                quiet=True,
                crop=(crop.x, crop.y, crop.w, crop.h),
                filters=filt,
                cancel_event=cancel_event,
            )
            path = execute(cfg)
        except JobCancelled:
            err = "cancelled"
            break
        except Exception as exc:  # noqa: BLE001 — collect per-variant failures
            err = str(exc)
        results.append(
            {
                "id": vid,
                "model": vid,  # back-compat for /api/compare/{id}/{model}
                "label": label,
                "kind": kind,
                "path": str(path) if path else None,
                "seconds": round(time.perf_counter() - t0, 2),
                "error": err,
            }
        )
        if err == "cancelled":
            break
    if progress:
        progress("compare done", 1.0)
    return results


def compare_models(
    image: Path,
    out_dir: Path,
    *,
    models: list[str],
    crop: CropRect | tuple[int, int, int, int],
    scale: float = 4.0,
    device: str = "auto",
    half: bool = False,
    progress: Callable[[str, float], None] | None = None,
) -> list[dict]:
    """Back-compat wrapper: neural models only."""
    variants = build_variants(models=models)
    return compare_variants(
        image,
        out_dir,
        variants=variants,
        crop=crop,
        scale=scale,
        device=device,
        half=half,
        progress=progress,
    )
