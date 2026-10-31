"""API routes split from server.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from upscaler.api import deps as D
from upscaler.models import (
    DEFAULT_WEIGHTS_DIR,
    add_external_model,
    ensure_model,
    model_status,
    validate_model_name,
)

router = APIRouter()

@router.get("/api/models")
def api_models_list() -> list[dict[str, Any]]:
    return [m.model_dump() for m in D._model_info_list()]

@router.post("/api/models")
def api_models_add(body: D.ExternalModelBody) -> dict[str, Any]:

    try:
        validate_model_name(body.name)
        entry = add_external_model(
            body.name,
            url=body.url,
            path=body.path,
            scale=body.scale,
            sha256=body.sha256,
            description=body.description,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"name": body.name, "entry": entry, "status": model_status(body.name)}

@router.post("/api/models/upload")
async def api_models_upload(
    file: UploadFile = File(...),
    name: str = Form(""),
    scale: int = Form(4),
    description: str = Form(""),
) -> dict[str, Any]:
    """Accept weights upload into the models directory and register it."""

    raw_name = (name or Path(file.filename or "model").stem).strip()
    try:
        validate_model_name(raw_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    suffix = Path(file.filename or "model.pth").suffix.lower() or ".pth"
    if suffix not in (".pth", ".pt", ".safetensors", ".onnx"):
        raise HTTPException(400, f"Unsupported weights extension: {suffix}")
    content = await file.read()
    if len(content) > D.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload exceeds limit of {D.MAX_UPLOAD_BYTES} bytes")
    DEFAULT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DEFAULT_WEIGHTS_DIR / f"{raw_name}{suffix}"
    dest.write_bytes(content)
    try:
        entry = add_external_model(
            raw_name,
            path=str(dest),
            scale=int(scale) if 1 <= int(scale) <= 8 else 4,
            description=description or f"Uploaded {file.filename}",
        )
    except (ValueError, FileNotFoundError) as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    return {
        "name": raw_name,
        "path": str(dest),
        "entry": entry,
        "status": model_status(raw_name),
    }

@router.post("/api/models/{name}/download")
def api_models_download(name: str) -> dict[str, Any]:
    """Ensure weights are on disk (download + hash verify)."""

    try:
        validate_model_name(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        path = ensure_model(name)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown model: {name}") from exc
    except Exception as exc:  # noqa: BLE001 — surface download errors to UI
        raise HTTPException(400, str(exc)) from exc
    return {"name": name, "path": str(path), "status": model_status(name)}
