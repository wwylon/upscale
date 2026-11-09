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
    if (js.width is None) ^ (js.height is None):
        raise HTTPException(400, "Specify both width and height, or neither.")
    # Soft output pixel budget (matches job_prep.MAX_OUTPUT_PIXELS intent).
    if js.width is not None and js.height is not None and js.width * js.height > 200_000_000:
        raise HTTPException(400, "Output dimensions exceed the pixel budget.")
    if js.scale is not None and js.scale > 64:
        raise HTTPException(400, "Scale exceeds the allowed maximum (64).")

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
        strip_path = job.job_dir / "checkpoints" / "strip.json"
        ck = D._strip_checkpoint_info(job)
        if ck is None:
            # Unusable leftover strip.json must not surface as "resume".
            if strip_path.is_file():
                D._discard_bad_checkpoint(job.job_dir)
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
            # Same validation resume-checkpoint uses (rejects incomplete carry, etc.).
            ck = D._strip_checkpoint_info_from_dir(d)
            if ck is None:
                D._discard_bad_checkpoint(d)
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
                    "checkpoint": ck,
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
    import time

    with D._jobs_lock:
        job = D._jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        job.cancel_event.set()
        # Queued jobs never reach the worker's cancel check until dequeued —
        # mark cancelled now so MAX_QUEUE slots free immediately.
        if job.state == D.JobState.queued:
            job.state = D.JobState.cancelled
            D._push_event(
                job, {"type": "cancelled", "pct": job.last_pct, "message": "cancelled"}
            )
            return {"status": "cancelled"}
        if job.state == D.JobState.paused:
            # Wake the pause wait so cancel is noticed between tiles.
            job.pause_event.clear()
            if job.pause_started_at is not None:
                job.paused_seconds += max(0.0, time.time() - job.pause_started_at)
                job.pause_started_at = None
    return {"status": "cancelling"}

@router.post("/api/jobs/{job_id}/pause")
def job_pause(job_id: str) -> dict[str, str]:
    import time

    with D._jobs_lock:
        job = D._jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if job.state not in (D.JobState.queued, D.JobState.running):
            raise HTTPException(
                409, f"Cannot pause job in state {job.state.value}"
            )
        job.pause_event.set()
        job.state = D.JobState.paused
        job.pause_started_at = time.time()
    D._push_event(job, {"type": "paused", "pct": job.last_pct, "message": "paused"})
    return {"status": "paused"}

@router.post("/api/jobs/{job_id}/resume")
def job_resume(job_id: str) -> dict[str, str]:
    import time

    with D._jobs_lock:
        job = D._jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if job.state != D.JobState.paused:
            raise HTTPException(
                409, f"Cannot resume job in state {job.state.value}"
            )
        job.pause_event.clear()
        if job.pause_started_at is not None:
            job.paused_seconds += max(0.0, time.time() - job.pause_started_at)
            job.pause_started_at = None
        job.state = D.JobState.running
    D._push_event(job, {"type": "resumed", "pct": job.last_pct, "message": "resumed"})
    return {"status": "running"}

@router.post("/api/jobs/{job_id}/resume-checkpoint")
def job_resume_checkpoint(job_id: str) -> dict[str, str]:
    """Re-queue a cancelled/errored job so strip tiling continues from flush_row."""
    with D._jobs_lock:
        job = D._jobs.get(job_id)
        if job is None:
            # Rehydrate outside would race; do it then re-enter.
            pass
    if job is None:
        job = D._rehydrate_job_from_disk(job_id)
    ck = D._strip_checkpoint_info(job)
    if ck is None:
        D._discard_bad_checkpoint(job.job_dir)
        raise HTTPException(400, "No strip checkpoint to resume")
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

    with D._jobs_lock:
        live = D._jobs.get(job_id) or job
        D._jobs[job_id] = live
        if live.state in (D.JobState.queued, D.JobState.running, D.JobState.paused):
            raise HTTPException(409, "Job is already active")
        if live.state not in (D.JobState.cancelled, D.JobState.error):
            raise HTTPException(
                409, f"Cannot resume-checkpoint from state {live.state.value}"
            )
        queued = sum(
            1
            for j in D._jobs.values()
            if j.state in (D.JobState.queued, D.JobState.running, D.JobState.paused)
        )
        if queued >= D.MAX_QUEUE:
            raise HTTPException(429, f"Job queue full (max {D.MAX_QUEUE})")
        live.cancel_event = threading.Event()
        live.pause_event = threading.Event()
        live.error = None
        live.output_path = None
        live.state = D.JobState.queued
        live.resume_from_checkpoint = True
        live.last_pct = float(ck.get("pct_approx") or 0)
        try:
            while True:
                live.events.get_nowait()
        except queue.Empty:
            pass
        job = live
    # Outside the lock — _ensure_worker also takes _jobs_lock (non-reentrant).
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

@router.get("/api/jobs/{job_id}/quality")
def job_quality(job_id: str, mode: str = "seam") -> dict[str, Any]:
    """Seam score, or Lanczos-baseline diff metrics, for a finished job.

    ``seam`` measures a vertical join on the result alone.
    ``diff`` compares the result with a Lanczos resize of the input
    (both capped so the check stays small) and writes ``quality_diff.png``.
    """
    mode_key = (mode or "seam").strip().lower()
    if mode_key not in ("seam", "diff"):
        raise HTTPException(400, "mode must be seam or diff")

    with D._jobs_lock:
        job = D._jobs.get(job_id)
    job_dir = job.job_dir if job is not None else D._safe_job_dir(job_id)
    result: Path | None = None
    input_path: Path | None = None
    if job is not None:
        input_path = job.input_path if job.input_path.is_file() else None
        if job.output_path is not None and job.output_path.is_file():
            result = job.output_path
    if result is None:
        result = D._find_result_file(job_dir)
    if input_path is None:
        inputs = sorted(job_dir.glob("input.*")) if job_dir.is_dir() else []
        input_path = inputs[0] if inputs else None
    if result is None or not result.is_file():
        raise HTTPException(404, "Result not ready")

    import numpy as np
    from PIL import Image

    from upscaler.io_utils import lanczos_resize, load_rgba
    from upscaler.quality import diff_metrics, strip_seam_score

    out_rgb, _, _ = load_rgba(result)
    if mode_key == "seam":
        height, width = out_rgb.shape[:2]
        if max(height, width) > 1024:
            scale = 1024 / max(height, width)
            out_rgb = lanczos_resize(
                out_rgb,
                max(2, round(width * scale)),
                max(2, round(height * scale)),
            )
            height, width = out_rgb.shape[:2]
        mid = width // 2
        if mid < 1 or width - mid < 1:
            score = 0.0
        else:
            score = strip_seam_score(out_rgb[:, :mid], out_rgb[:, mid:], overlap=1)
        return {
            "mode": "seam",
            "seam_score": score,
            "width": int(width),
            "height": int(height),
        }

    if input_path is None or not input_path.is_file():
        raise HTTPException(404, "Input not found")
    in_rgb, _, _ = load_rgba(input_path)
    out_h, out_w = out_rgb.shape[:2]
    cap = 512
    if max(out_h, out_w) > cap:
        scale = cap / max(out_h, out_w)
        tw, th = max(1, round(out_w * scale)), max(1, round(out_h * scale))
        compared = lanczos_resize(out_rgb, tw, th)
        preview = True
    else:
        tw, th = out_w, out_h
        compared = out_rgb
        preview = False
    baseline = lanczos_resize(in_rgb, tw, th)
    metrics = diff_metrics(compared, baseline)
    delta = np.abs(compared.astype(np.int16) - baseline.astype(np.int16))
    vis = np.clip(delta * 4, 0, 255).astype(np.uint8)
    png_path = job_dir / "quality_diff.png"
    Image.fromarray(vis, mode="RGB").save(png_path)
    return {
        "mode": "diff",
        "metrics": metrics,
        "preview": preview,
        "width": int(tw),
        "height": int(th),
        "png": png_path.name,
    }

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
