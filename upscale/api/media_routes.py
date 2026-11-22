"""Estimate + decode-preview routes (split from server.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from upscale.api import deps as D
from upscale.schema import FilterSettings

router = APIRouter()


class EstimateBody(BaseModel):
    src_w: int = Field(gt=0)
    src_h: int = Field(gt=0)
    scale: float = Field(gt=0)
    model: str = "RealESRGAN_x4plus"
    device: str = "auto"
    model_ready: bool | None = None
    filters: FilterSettings | None = None
    measured_mp_s: float | None = None
    tile: int | None = None
    half: bool = False
    algorithm: str | None = None


@router.post("/api/estimate")
def api_estimate(body: EstimateBody) -> dict[str, Any]:
    """Honest time range; prefers learned MP/s from past jobs on this machine."""
    from upscale.estimates import estimate_time_seconds, model_fits_vram
    from upscale.filters import estimate_time_factor
    from upscale.models import DEFAULT_WEIGHTS_DIR, MODELS, all_models, model_status, registry_scale
    from upscale.speed_stats import lookup_measured_mp_s, normalize_device, work_megapixels
    from upscale.system import gather_system_info

    name = body.model
    ready = body.model_ready
    if ready is None:
        ready = model_status(name) == "installed"
    meta = all_models().get(name) or MODELS.get(name) or {}
    model_bytes = meta.get("size_bytes")
    if model_bytes is None and ready:
        url_path = (meta.get("url") or "").split("?")[0]
        ext = Path(url_path).suffix or ".pth"
        dest = DEFAULT_WEIGHTS_DIR / f"{name}{ext}"
        if dest.is_file():
            model_bytes = dest.stat().st_size
    mscale = int(meta.get("scale") or registry_scale(name) or 4)
    filt = body.filters or FilterSettings()
    factor = estimate_time_factor(filt)
    sys_info = gather_system_info(device=body.device, fp16=body.half)
    device = normalize_device(sys_info.device or body.device)
    work_mp = work_megapixels(
        body.src_w,
        body.src_h,
        body.scale,
        model_scale=mscale,
        algorithm=body.algorithm,
    )
    measured = body.measured_mp_s
    calibrated = False
    samples = 0
    stats_key = None
    if measured is None:
        hit = lookup_measured_mp_s(
            model=name,
            device=device,
            half=body.half,
            algorithm=body.algorithm,
            work_mp=work_mp,
        )
        if hit is not None:
            measured = float(hit["mp_s"])
            calibrated = True
            samples = int(hit["n"])
            stats_key = hit.get("key")
    elif measured > 0:
        calibrated = True
    lo, hi = estimate_time_seconds(
        body.src_w,
        body.src_h,
        body.scale,
        model=name,
        device=device,
        model_scale=mscale,
        model_ready=bool(ready),
        model_bytes=model_bytes,
        filter_factor=factor,
        measured_mp_s=measured,
        algorithm=body.algorithm,
        calibrated=calibrated,
    )
    tile = body.tile or sys_info.auto_tile or 512
    vram_fit = model_fits_vram(
        model_bytes,
        sys_info.vram_free_mb,
        tile=int(tile),
        fp16=body.half,
    )
    return {
        "seconds_low": lo,
        "seconds_high": hi,
        "seconds": round((lo + hi) / 2, 1),
        "model_ready": bool(ready),
        "filter_factor": factor,
        "vram_fit": vram_fit,
        "vram_free_mb": sys_info.vram_free_mb,
        "tile": int(tile),
        "calibrated": calibrated,
        "samples": samples,
        "mp_s": measured,
        "device": device,
        "work_mp": round(work_mp, 4),
        "stats_key": stats_key,
    }


class PreflightBody(BaseModel):
    src_w: int = Field(gt=0)
    src_h: int = Field(gt=0)
    scale: float | None = None
    target_size: int | None = None
    width: int | None = None
    height: int | None = None
    model: str = "RealESRGAN_x4plus"
    device: str = "auto"
    half: bool = False
    tile: int | None = None
    algorithm: str | None = None


@router.post("/api/preflight")
def api_preflight(body: PreflightBody) -> dict[str, Any]:
    """Structured sizing + resource check before enqueueing a job."""
    import click

    from upscale.config import UpscaleConfig
    from upscale.estimates import resource_check
    from upscale.job_prep import resolve_scale_and_targets
    from upscale.models import DEFAULT_WEIGHTS_DIR, MODELS, all_models, model_status, registry_scale
    from upscale.system import gather_system_info

    cfg = UpscaleConfig(
        scale=body.scale,
        target_size=body.target_size,
        width=body.width,
        height=body.height,
    )
    if cfg.scale is None and cfg.target_size is None and (
        cfg.width is None or cfg.height is None
    ):
        cfg.scale = 4.0
    try:
        scale, tw, th = resolve_scale_and_targets(cfg, body.src_w, body.src_h)
    except click.UsageError as exc:
        raise HTTPException(400, str(exc)) from exc

    name = body.model
    ready = model_status(name) == "installed" or body.algorithm is not None
    meta = all_models().get(name) or MODELS.get(name) or {}
    model_bytes = meta.get("size_bytes")
    if model_bytes is None and ready and body.algorithm is None:
        url_path = (meta.get("url") or "").split("?")[0]
        ext = Path(url_path).suffix or ".pth"
        dest = DEFAULT_WEIGHTS_DIR / f"{name}{ext}"
        if dest.is_file():
            model_bytes = dest.stat().st_size
    sys_info = gather_system_info(device=body.device, fp16=body.half)
    tile = int(body.tile or sys_info.auto_tile or 512)
    check = resource_check(
        out_w=tw,
        out_h=th,
        tile=tile,
        fp16=body.half,
        model_bytes=None if body.algorithm else model_bytes,
        vram_free_mb=sys_info.vram_free_mb,
        ram_available_mb=sys_info.ram_available_mb,
        disk_free_mb=sys_info.disk_free_mb,
        model_ready=bool(ready),
    )
    return {
        "ok": not check.blocked,
        "blocked": check.blocked,
        "scale": scale,
        "target_w": tw,
        "target_h": th,
        "tile": tile,
        "model_scale": int(meta.get("scale") or registry_scale(name) or 4),
        "checks": [
            {
                "name": r.name,
                "status": r.status.value if hasattr(r.status, "value") else str(r.status),
                "detail": r.detail,
                "advice": getattr(r, "advice", "") or "",
            }
            for r in check.rows
        ],
    }


@router.post("/api/decode-preview")
async def decode_preview(file: UploadFile = File(...)) -> Response:
    """Decode any supported input format to a browser-safe PNG preview."""
    import io
    import tempfile

    import numpy as np
    from PIL import Image

    from upscale.io_utils import is_vector_input, load_rgba, require_raster_suffix

    try:
        suffix = require_raster_suffix(file.filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if file.content_type and is_vector_input(file.filename, file.content_type):
        raise HTTPException(
            400,
            "Vector images are not supported. "
            "Rasterize to PNG, TIFF, or another supported bitmap format first.",
        )
    content = await file.read()
    if len(content) > D.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload exceeds limit of {D.MAX_UPLOAD_BYTES} bytes")
    with tempfile.TemporaryDirectory(prefix="upscale-decode-") as td:
        src = Path(td) / f"input{suffix}"
        src.write_bytes(content)
        try:
            rgb, alpha, _mode = load_rgba(src)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Could not decode image: {exc}") from exc

        if alpha is not None:
            im = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
        else:
            im = Image.fromarray(rgb, mode="RGB")
        h, w = rgb.shape[:2]
        edge = 4096
        if max(im.size) > edge:
            im = im.copy()
            im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=False)
        data = buf.getvalue()
    headers = {
        "X-Image-Width": str(w),
        "X-Image-Height": str(h),
        "X-Has-Alpha": "1" if alpha is not None else "0",
        "X-Source-Name": Path(file.filename or "image").name,
        "Cache-Control": "no-store",
    }
    return Response(content=data, media_type="image/png", headers=headers)
