"""API routes split from server.py."""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from upscaler.api import deps as D
from upscaler.schema import JobSettings

router = APIRouter()

@router.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    settings: str = Form("{}"),
) -> dict[str, str]:
    D._purge_expired_jobs()
    try:
        raw = json.loads(settings) if settings.strip() else {}
        js = JobSettings.model_validate(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid settings JSON: {exc}") from exc

    if js.scale is None and js.target_size is None and (js.width is None or js.height is None):
        js = js.model_copy(update={"scale": 4.0})

    with D._jobs_lock:
        queued = sum(
            1
            for j in D._jobs.values()
            if j.state in (D.JobState.queued, D.JobState.running, D.JobState.paused)
        )
    if queued >= D.MAX_QUEUE:
        raise HTTPException(429, f"Job queue full (max {D.MAX_QUEUE})")

    job_id = uuid.uuid4().hex
    job_dir = D.JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    from upscaler.io_utils import is_vector_input, require_raster_suffix

    try:
        suffix = require_raster_suffix(file.filename)
    except ValueError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc
    if is_vector_input(file.filename, file.content_type):
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            400,
            "Vector images are not supported. "
            "Rasterize to PNG, TIFF, or another supported bitmap format first.",
        )
    input_path = job_dir / f"input{suffix}"
    content = await file.read()
    if len(content) > D.MAX_UPLOAD_BYTES:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            413, f"Upload exceeds limit of {D.MAX_UPLOAD_BYTES} bytes"
        )
    input_path.write_bytes(content)
    try:
        D._make_preview(input_path, job_dir / "preview.png")
    except Exception:  # noqa: BLE001
        try:
            shutil.copy(input_path, job_dir / "preview.png")
        except Exception:  # noqa: BLE001
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, "Could not read uploaded image") from None

    job = D.JobRecord(id=job_id, job_dir=job_dir, input_path=input_path, settings=js)
    (job_dir / "settings.json").write_text(
        js.model_dump_json(exclude_none=True), encoding="utf-8"
    )
    with D._jobs_lock:
        D._jobs[job_id] = job
    D._ensure_worker()
    D._job_queue.put(job_id)
    D._push_event(job, {"type": "queued", "pct": 0, "message": "queued", "id": job_id})
    return {"id": job_id, "input": str(input_path)}

@router.get("/api/jobs/interrupted")
def list_interrupted_jobs() -> dict[str, Any]:
    """Jobs that still have a strip checkpoint (cancelled/error mid-pass).

    Registered before ``/api/jobs/{job_id}`` so ``interrupted`` is not captured as an id.
    Skips active and completed jobs; clears leftover checkpoints on done jobs and
    disk dirs that already have a result. Skips unreadable inputs.
    """
    D._purge_expired_jobs()
    items: list[dict[str, Any]] = []
    with D._jobs_lock:
        jobs = list(D._jobs.values())
    known: set[str] = set()
    for job in jobs:
        known.add(job.id)
        if job.state in (D.JobState.queued, D.JobState.running, D.JobState.paused):
            continue
        ck = D._strip_checkpoint_info(job)
        if ck is None:
            continue
        # Finished jobs must not offer "resume" — clear stale strip markers.
        if job.state == D.JobState.done:
            D._discard_bad_checkpoint(job.job_dir)
            continue
        if job.state not in (D.JobState.cancelled, D.JobState.error):
            continue
        if not job.input_path.is_file() or not D._png_readable(job.input_path):
            D._discard_bad_checkpoint(job.job_dir)
            continue
        items.append(
            {
                "id": job.id,
                "state": job.state.value,
                "checkpoint": ck,
                "settings": job.settings.model_dump(mode="json", exclude_none=True),
                "input": job.input_path.name,
                "mtime": job.job_dir.stat().st_mtime if job.job_dir.is_dir() else 0,
            }
        )
    if D.JOBS_ROOT.is_dir():
        for d in D.JOBS_ROOT.iterdir():
            if not d.is_dir() or d.name in known or d.name.startswith("compare-"):
                continue
            strip = d / "checkpoints" / "strip.json"
            if not strip.is_file():
                continue
            # Already finished on disk — not interrupted.
            if D._find_result_file(d) is not None:
                D._discard_bad_checkpoint(d)
                continue
            try:
                raw = json.loads(strip.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            inputs = sorted(d.glob("input.*"))
            if not inputs or not D._png_readable(inputs[0]):
                D._discard_bad_checkpoint(d)
                continue
            settings = None
            sp = d / "settings.json"
            if sp.is_file():
                try:
                    settings = json.loads(sp.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    settings = None
            items.append(
                {
                    "id": d.name,
                    "state": "orphaned",
                    "checkpoint": {
                        "pass_index": raw.get("pass_index"),
                        "flush_row": raw.get("flush_row"),
                        "out_h": raw.get("out_h"),
                        "out_w": raw.get("out_w"),
                    },
                    "settings": settings,
                    "input": inputs[0].name,
                    "mtime": d.stat().st_mtime,
                }
            )
    items.sort(key=lambda j: float(j.get("mtime") or 0), reverse=True)
    for j in items:
        j.pop("mtime", None)
    return {"jobs": items}

@router.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    with D._jobs_lock:
        job = D._jobs.get(job_id)
    if job is not None:
        return {
            "id": job.id,
            "state": job.state.value,
            "pct": job.last_pct,
            "error": job.error,
            "cli": job.cli_hint,
            "output": str(job.output_path) if job.output_path else None,
            "input": str(job.input_path),
            "created_at": job.created_at,
        }
    # Server restart: still report done if artifacts remain on disk
    job_dir = D._safe_job_dir(job_id)
    display = job_dir / "display.png"
    result = D._find_result_file(job_dir)
    if display.is_file() or (result is not None and result.is_file()):
        return {
            "id": job_id,
            "state": D.JobState.done.value,
            "pct": 1.0,
            "error": None,
            "cli": "",
            "output": str(result) if result is not None else None,
            "created_at": job_dir.stat().st_mtime,
        }
    raise HTTPException(404, "Job not found")

@router.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    job = D._get_job(job_id)

    async def stream() -> Any:
        while True:
            try:
                ev = await asyncio.to_thread(job.events.get, True, 30.0)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            payload = json.dumps(ev)
            yield f"data: {payload}\n\n"
            if ev.get("type") in ("done", "error", "cancelled"):
                break

    return StreamingResponse(stream(), media_type="text/event-stream")

@router.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict[str, str]:
    job = D._get_job(job_id)
    job.cancel_event.set()
    return {"status": "cancelling"}

@router.post("/api/jobs/{job_id}/pause")
def job_pause(job_id: str) -> dict[str, str]:
    job = D._get_job(job_id)
    job.pause_event.set()
    job.state = D.JobState.paused
    D._push_event(job, {"type": "paused", "pct": job.last_pct, "message": "paused"})
    return {"status": "paused"}

@router.post("/api/jobs/{job_id}/resume")
def job_resume(job_id: str) -> dict[str, str]:
    import time

    job = D._get_job(job_id)
    job.pause_event.clear()
    if job.pause_started_at is not None:
        job.paused_seconds += max(0.0, time.time() - job.pause_started_at)
        job.pause_started_at = None
    if job.state == D.JobState.paused:
        job.state = D.JobState.running
    D._push_event(job, {"type": "resumed", "pct": job.last_pct, "message": "resumed"})
    return {"status": "running"}

@router.post("/api/jobs/{job_id}/resume-checkpoint")
def job_resume_checkpoint(job_id: str) -> dict[str, str]:
    """Re-queue a cancelled/errored job so strip tiling continues from flush_row."""
    with D._jobs_lock:
        job = D._jobs.get(job_id)
    if job is None:
        job = D._rehydrate_job_from_disk(job_id)
    ck = D._strip_checkpoint_info(job)
    if ck is None:
        raise HTTPException(400, "No strip checkpoint to resume")
    if job.state in (D.JobState.queued, D.JobState.running, D.JobState.paused):
        raise HTTPException(409, "Job is already active")
    if not job.input_path.is_file():
        D._discard_bad_checkpoint(job.job_dir)
        raise HTTPException(400, "Input file missing for this job")
    if not D._png_readable(job.input_path):
        D._discard_bad_checkpoint(job.job_dir)
        raise HTTPException(
            400,
            "Input image is corrupt or unreadable; checkpoint cleared. "
            "Start a new upscale instead.",
        )

    job.cancel_event = threading.Event()
    job.pause_event = threading.Event()
    job.error = None
    job.output_path = None
    job.state = D.JobState.queued
    job.resume_from_checkpoint = True
    job.last_pct = float(ck.get("pct_approx") or 0)
    try:
        while True:
            job.events.get_nowait()
    except queue.Empty:
        pass
    D._ensure_worker()
    D._job_queue.put(job_id)
    D._push_event(
        job,
        {
            "type": "queued",
            "pct": job.last_pct,
            "message": "resuming from checkpoint",
            "id": job_id,
        },
    )
    return {"status": "queued", "id": job_id}

@router.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> FileResponse:
    with D._jobs_lock:
        job = D._jobs.get(job_id)
    if job is not None and job.output_path is not None and job.output_path.is_file():
        return FileResponse(job.output_path, filename=job.output_path.name)
    result = D._find_result_file(D._safe_job_dir(job_id))
    if result is not None and result.is_file():
        return FileResponse(result, filename=result.name)
    raise HTTPException(404, "Result not ready")

@router.get("/api/jobs/{job_id}/preview")
def job_preview(job_id: str) -> FileResponse:
    """Small input thumbnail (≤512 px)."""
    with D._jobs_lock:
        job = D._jobs.get(job_id)
    preview = (job.job_dir if job is not None else D._safe_job_dir(job_id)) / "preview.png"
    if not preview.is_file():
        # Fall back to input.*
        job_dir = preview.parent
        inputs = sorted(job_dir.glob("input.*"))
        if inputs:
            try:
                D._make_preview(inputs[0], preview, max_edge=512)
            except Exception:  # noqa: BLE001
                pass
    if not preview.is_file():
        raise HTTPException(404, "Preview not found")
    return FileResponse(preview, media_type="image/png")

@router.get("/api/jobs/{job_id}/display")
def job_display(job_id: str) -> FileResponse:
    """Browser-safe result preview (long edge ≤4096). Full file is /result.

    Serves from disk even after process restart (job no longer in memory).
    """
    job_dir = D._safe_job_dir(job_id)
    display = job_dir / "display.png"
    if display.is_file():
        return FileResponse(display, media_type="image/png")

    with D._jobs_lock:
        job = D._jobs.get(job_id)
    result: Path | None = None
    if job is not None and job.output_path is not None and job.output_path.is_file():
        result = job.output_path
    else:
        result = D._find_result_file(job_dir)

    if result is not None and result.is_file():
        try:
            D._make_preview(result, display, max_edge=4096)
            if display.is_file():
                return FileResponse(display, media_type="image/png")
        except Exception:  # noqa: BLE001
            pass
        return FileResponse(result, filename=result.name)
    raise HTTPException(404, "Display preview not ready")

@router.get("/api/jobs/{job_id}/passes/{pass_n:int}")
def job_pass(job_id: str, pass_n: int) -> FileResponse:
    job_dir = D._safe_job_dir(job_id)
    art = job_dir / f"pass_{pass_n}.png"
    if not art.is_file():
        raise HTTPException(404, "Pass preview not found")
    return FileResponse(art, media_type="image/png")
