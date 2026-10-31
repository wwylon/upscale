"""API routes split from server.py."""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from upscaler.api import deps as D
from upscaler.tiling import JobCancelled

router = APIRouter()

class ZipBody(BaseModel):
    job_ids: list[str]


@router.post("/api/zip")
def api_zip(body: ZipBody) -> Response:
    """Zip completed job result files for batch download."""
    import io
    import zipfile

    if not body.job_ids:
        raise HTTPException(400, "job_ids required")
    if len(body.job_ids) > 64:
        raise HTTPException(400, "too many jobs")
    buf = io.BytesIO()
    added = 0
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for jid in body.job_ids:
            result: Path | None = None
            with D._jobs_lock:
                job = D._jobs.get(jid)
            if job is not None and job.output_path is not None and job.output_path.is_file():
                result = job.output_path
            else:
                try:
                    result = D._find_result_file(D._safe_job_dir(jid))
                except HTTPException:
                    continue
            if result is None or not result.is_file():
                continue
            arc = f"{jid}_{result.name}"
            zf.write(result, arcname=arc)
            added += 1
    if added == 0:
        raise HTTPException(404, "No completed results to zip")
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="upscale-batch.zip"'},
    )

class CompareBody(BaseModel):
    models: list[str] = Field(default_factory=list)
    algorithms: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    filter_amounts: dict[str, float] = Field(default_factory=dict)
    base_model: str = "RealESRGAN_x4plus"
    base_algorithm: str | None = None
    crop: dict[str, int]
    scale: float = 4.0
    device: str = "auto"
    half: bool = False


@dataclass
class CompareRun:
    id: str
    dir: Path
    cancel_event: threading.Event = field(default_factory=threading.Event)
    state: str = "running"  # running | done | cancelled | error
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    message: str = ""
    created_at: float = field(default_factory=lambda: __import__("time").time())




def _compare_worker(
    run: CompareRun,
    input_path: Path,
    variants: list[dict[str, Any]],
    crop: Any,
    scale: float,
    device: str,
    half: bool,
) -> None:
    from upscaler.compare import compare_variants

    try:
        # Share the process-wide execute lock so compare never overlaps a job.
        with D._execute_lock:
            results = compare_variants(
                input_path,
                run.dir,
                variants=variants,
                crop=crop,
                scale=scale,
                device=device,
                half=half,
                cancel_event=run.cancel_event,
                progress=lambda msg, _frac: setattr(run, "message", msg),
            )
        run.results = results
        # Persist results for status after in-memory eviction
        try:
            (run.dir / "compare_results.json").write_text(
                json.dumps(results), encoding="utf-8"
            )
        except OSError:
            pass
        if run.cancel_event.is_set():
            run.state = "cancelled"
            run.message = "cancelled"
        else:
            run.state = "done"
            run.message = ""
    except JobCancelled:
        run.state = "cancelled"
        run.message = "cancelled"
    except Exception as exc:  # noqa: BLE001 — surface to status poller
        run.state = "error"
        run.error = str(exc)
        run.message = str(exc)


@router.post("/api/compare")
async def api_compare(
    file: UploadFile = File(...),
    body: str = Form("{}"),
) -> dict[str, Any]:
    """Start a crop compare; poll ``GET /api/compare/{id}/status`` until done."""
    from upscaler.compare import build_variants
    from upscaler.schema import CropRect

    try:
        raw = json.loads(body) if body.strip() else {}
        cb = CompareBody.model_validate(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid compare body: {exc}") from exc
    variants = build_variants(
        models=cb.models,
        algorithms=cb.algorithms,
        filters=cb.filters,
        filter_amounts=cb.filter_amounts,
        base_model=cb.base_model,
        base_algorithm=cb.base_algorithm,
    )
    if not variants:
        raise HTTPException(
            400, "Select at least one model, classical algorithm, or filter"
        )

    run_id = uuid.uuid4().hex
    run_dir = D.JOBS_ROOT / f"compare-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "input.png").suffix or ".png"
    input_path = run_dir / f"input{suffix}"
    content = await file.read()
    if len(content) > D.MAX_UPLOAD_BYTES:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(413, f"Upload exceeds limit of {D.MAX_UPLOAD_BYTES} bytes")
    input_path.write_bytes(content)
    crop = CropRect(
        x=int(cb.crop.get("x", 0)),
        y=int(cb.crop.get("y", 0)),
        w=int(cb.crop.get("w", 64)),
        h=int(cb.crop.get("h", 64)),
    )
    run = CompareRun(id=run_id, dir=run_dir, message="starting")
    with D._compare_lock:
        D._compare_jobs[run_id] = run
        D._compare_runs[run_id] = run_dir
    threading.Thread(
        target=_compare_worker,
        args=(run, input_path, variants, crop, cb.scale, cb.device, cb.half),
        name=f"compare-{run_id[:8]}",
        daemon=True,
    ).start()
    return {"id": run_id, "state": "running"}

@router.get("/api/compare/{run_id}/status")
def api_compare_status(run_id: str) -> dict[str, Any]:
    with D._compare_lock:
        run = D._compare_jobs.get(run_id)
    if run is None:
        run_dir = D._compare_runs.get(run_id) or (D.JOBS_ROOT / f"compare-{run_id}")
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise HTTPException(404, "Compare run not found")
        results: list[dict[str, Any]] = []
        meta = run_dir / "compare_results.json"
        if meta.is_file():
            try:
                results = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                results = []
        return {
            "id": run_id,
            "state": "done" if results else "error",
            "results": results,
            "message": "" if results else "Compare results unavailable after restart",
            "error": None if results else "results missing",
        }
    return {
        "id": run.id,
        "state": run.state,
        "results": run.results,
        "error": run.error,
        "message": run.message,
    }

@router.post("/api/compare/{run_id}/cancel")
def api_compare_cancel(run_id: str) -> dict[str, str]:
    with D._compare_lock:
        run = D._compare_jobs.get(run_id)
    if run is None:
        raise HTTPException(404, "Compare run not found")
    run.cancel_event.set()
    run.message = "cancelling"
    return {"status": "cancelling"}

@router.get("/api/compare/{run_id}/{model:path}")
def api_compare_result(run_id: str, model: str) -> FileResponse:
    run_dir = D._compare_runs.get(run_id)
    if run_dir is None:
        run_dir = D.JOBS_ROOT / f"compare-{run_id}"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)
    path = run_dir / f"compare_{safe}.png"
    if not path.is_file():
        raise HTTPException(404, "Compare result not found")
    return FileResponse(path, media_type="image/png")
