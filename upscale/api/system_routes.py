"""API routes split from server.py."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from upscale.api import deps as D
from upscale.estimates import model_fits_vram
from upscale.help_content import get_help, list_topics
from upscale.models import DEFAULT_WEIGHTS_DIR, MODELS, all_models
from upscale.schema import LANGUAGE_CATALOG
from upscale.system import gather_system_info

router = APIRouter()

@router.get("/api/algorithms")
def api_algorithms_list() -> dict[str, Any]:
    from dataclasses import asdict

    from upscale.algorithms import list_algorithm_meta, list_algorithms
    from upscale.algorithms.classical import backend_name

    meta = list_algorithm_meta()
    return {
        "algorithms": list_algorithms(),
        "items": [asdict(m) for m in meta],
        "backend": backend_name(),
    }

@router.get("/api/system")
def api_system(
    device: str = "auto",
    half: bool = False,
    model: str | None = None,
    tile: int | None = None,
) -> dict[str, Any]:
    info = gather_system_info(device=device, fp16=half)
    payload = info.model_dump()
    # Optional preflight VRAM fit for a chosen model + tile
    model_bytes = None
    if model:
        meta = all_models().get(model) or MODELS.get(model) or {}
        model_bytes = meta.get("size_bytes")
        if model_bytes is None:
            url_path = (meta.get("url") or "").split("?")[0]
            ext = Path(url_path).suffix or ".pth"
            dest = DEFAULT_WEIGHTS_DIR / f"{model}{ext}"
            if dest.is_file():
                model_bytes = dest.stat().st_size
    use_tile = tile or info.auto_tile or 512
    payload["vram_fit"] = model_fits_vram(
        model_bytes,
        info.vram_free_mb,
        tile=int(use_tile),
        fp16=half,
    )
    payload["vram_fit_tile"] = int(use_tile)
    return payload

@router.get("/api/about")
def api_about() -> dict[str, Any]:
    from upscale.about import about_payload, models_disk_summary

    payload = about_payload()
    payload["models"] = models_disk_summary()
    return payload

@router.get("/api/license")
def api_license() -> dict[str, Any]:
    from upscale.about import license_text, package_version

    return {"license": "MIT", "version": package_version(), "text": license_text()}

@router.get("/api/updates")
def api_updates() -> dict[str, Any]:
    from upscale.about import check_for_updates

    return check_for_updates()

@router.get("/api/help")
def api_help_index(lang: str = "en") -> dict[str, Any]:
    return {"topics": list_topics(lang), "languages": LANGUAGE_CATALOG}

@router.get("/api/help/{topic}")
def api_help_topic(topic: str, lang: str = "en") -> dict[str, str]:
    try:
        body = get_help(topic, lang)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"topic": topic, "lang": lang, "markdown": body}

@router.get("/api/presets")
def api_presets() -> dict[str, Any]:
    from upscale.presets import list_presets

    return list_presets()

class PresetApplyBody(BaseModel):
    id: str
    kind: str = "option"  # option | target


@router.post("/api/presets/apply")
def api_presets_apply(body: PresetApplyBody) -> dict[str, Any]:
    from upscale.presets import _target_settings, resolve_option_preset, resolve_target_preset

    try:
        if body.kind == "target":
            target = resolve_target_preset(body.id)
            return {
                "id": target.id,
                "kind": "target",
                "label": target.label,
                "settings": _target_settings(target),
            }
        option = resolve_option_preset(body.id)
        return {
            "id": option.id,
            "kind": "option",
            "label": option.label,
            "settings": dict(option.settings),
        }
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

@router.post("/api/suggest")
async def api_suggest(
    file: UploadFile = File(...),
    fast: bool = Form(False),
) -> dict[str, Any]:
    from upscale.suggest import suggest_model

    job_dir = Path(tempfile.mkdtemp(prefix="upscale-suggest-"))
    try:
        suffix = Path(file.filename or "input.png").suffix or ".png"
        dest = job_dir / f"input{suffix}"
        content = await file.read()
        if len(content) > D.MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"Upload exceeds limit of {D.MAX_UPLOAD_BYTES} bytes"
            )
        dest.write_bytes(content)
        return suggest_model(dest, fast=fast)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

class BenchmarkBody(BaseModel):
    path: str
    model: str = "RealESRGAN_x4plus"
    device: str = "cpu"
    crop_size: int = 64
    scale: float = 2.0


@router.post("/api/benchmark")
def api_benchmark(body: BenchmarkBody) -> dict[str, Any]:
    from upscale.benchmark import run_benchmark

    p = Path(body.path).resolve()
    if not p.is_file():
        raise HTTPException(400, "path must be a readable local file")
    # Sandbox: only allow files under D.JOBS_ROOT (or system temp upscale-jobs).
    jobs_root = D.JOBS_ROOT.resolve()
    if not p.is_relative_to(jobs_root):
        raise HTTPException(
            400,
            "benchmark path must be under the jobs temp directory "
            f"({jobs_root})",
        )
    with D._execute_lock:
        return run_benchmark(
            p,
            model=body.model,
            device=body.device,
            crop_size=body.crop_size,
            scale=body.scale,
        )

@router.get("/api/locales/{code}")
def api_locale(code: str) -> JSONResponse:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", code or ""):
        raise HTTPException(404, "Locale not found")
    locales = D.LOCALES_DIR.resolve()
    path = (locales / f"{code}.json").resolve()
    if not path.is_relative_to(locales):
        raise HTTPException(404, "Locale not found")
    if not path.is_file():
        path = (locales / "en.json").resolve()
        if not path.is_file() or not path.is_relative_to(locales):
            raise HTTPException(404, "Locale not found")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
