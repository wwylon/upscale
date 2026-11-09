"""FastAPI job API and static web UI."""

from __future__ import annotations

import queue
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
    Response,
)
from pydantic import BaseModel, Field

from .cli import settings_to_cli
from .config import UpscaleConfig
from .engine import execute
from .estimates import estimate_time_seconds, model_fits_vram
from .filters import estimate_time_factor
from .models import (
    DEFAULT_WEIGHTS_DIR,
    MODELS,
    all_models,
    model_status,
    registry_scale,
)
from .schema import LANGUAGE_CATALOG, FilterSettings, JobSettings, ModelInfo
from .system import gather_system_info
from .tiling import JobCancelled


def _env_path(name: str, default: Path) -> Path:
    raw = __import__("os").environ.get(name)
    return Path(raw) if raw else default


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = _env_path("UPSCALE_UI_DIR", PROJECT_ROOT / "ui")
LOCALES_DIR = _env_path("UPSCALE_LOCALES_DIR", PROJECT_ROOT / "locales")
DOCS_DIR = _env_path("UPSCALE_DOCS_DIR", PROJECT_ROOT / "docs")
def _default_jobs_root() -> Path:
    """Prefer a per-user app data dir over the shared system temp."""
    import os

    override = os.environ.get("UPSCALE_JOBS_DIR")
    if override:
        return Path(override)
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if local:
        return Path(local) / "Upscale" / "jobs"
    return Path(tempfile.gettempdir()) / "upscale-jobs"


JOBS_ROOT = _default_jobs_root()
JOB_TTL_S = int(__import__("os").environ.get("UPSCALE_JOB_TTL_S", str(6 * 3600)))
MAX_JOBS = int(__import__("os").environ.get("UPSCALE_MAX_JOBS", "32"))
MAX_UPLOAD_BYTES = int(
    __import__("os").environ.get("UPSCALE_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024))
)
MAX_QUEUE = int(__import__("os").environ.get("UPSCALE_MAX_QUEUE", "16"))

# Desktop / frozen builds store models under %LOCALAPPDATA%\Upscale\models
_models_override = __import__("os").environ.get("UPSCALE_MODELS_DIR")
if _models_override:
    from . import models as _models_mod

    _models_mod.DEFAULT_WEIGHTS_DIR = Path(_models_override)

app = FastAPI(title="Upscale", version="0.4.0")


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    paused = "paused"
    done = "done"
    error = "error"
    cancelled = "cancelled"


@dataclass
class JobRecord:
    id: str
    job_dir: Path
    input_path: Path
    settings: JobSettings
    state: JobState = JobState.queued
    events: queue.Queue[dict[str, Any]] = field(default_factory=queue.Queue)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    output_path: Path | None = None
    error: str | None = None
    cli_hint: str = ""
    created_at: float = field(default_factory=lambda: __import__("time").time())
    last_pct: float = 0.0
    # Wall-clock accounting for speed learning (excludes pause time).
    run_started_at: float | None = None
    pause_started_at: float | None = None
    paused_seconds: float = 0.0
    resume_from_checkpoint: bool = False


_jobs: dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()
_job_queue: queue.Queue[str | None] = queue.Queue()
_worker_started = False
# Serialize GPU/CPU inference across job worker and compare threads.
_execute_lock = threading.Lock()


def _purge_expired_compare_jobs() -> None:
    import time

    now = time.time()
    with _compare_lock:
        dead = [
            rid
            for rid, run in _compare_jobs.items()
            if run.state in ("done", "error", "cancelled")
            and now - getattr(run, "created_at", now) > JOB_TTL_S
        ]
        for rid in dead:
            run = _compare_jobs.pop(rid, None)
            _compare_runs.pop(rid, None)
            if run is not None:
                shutil.rmtree(run.dir, ignore_errors=True)


def _purge_stale_job_dirs_on_disk() -> None:
    """Delete job directories older than JOB_TTL_S that have no valid strip CK."""
    import time

    now = time.time()
    if not JOBS_ROOT.is_dir():
        return
    for child in JOBS_ROOT.iterdir():
        if not child.is_dir():
            continue
        try:
            age = now - child.stat().st_mtime
        except OSError:
            continue
        if age <= JOB_TTL_S:
            continue
        strip = child / "checkpoints" / "strip.json"
        if strip.is_file():
            continue
        shutil.rmtree(child, ignore_errors=True)


def _purge_expired_jobs() -> None:
    import time

    now = time.time()
    _purge_expired_compare_jobs()
    with _jobs_lock:
        dead = [
            jid
            for jid, j in _jobs.items()
            if j.state in (JobState.done, JobState.error, JobState.cancelled)
            and now - j.created_at > JOB_TTL_S
        ]
        for jid in dead:
            job = _jobs.pop(jid, None)
            if job is not None:
                shutil.rmtree(job.job_dir, ignore_errors=True)
        # Cap total retained jobs (oldest finished first)
        if len(_jobs) > MAX_JOBS:
            finished = sorted(
                (
                    (jid, j)
                    for jid, j in _jobs.items()
                    if j.state
                    in (JobState.done, JobState.error, JobState.cancelled)
                ),
                key=lambda x: x[1].created_at,
            )
            overflow = len(_jobs) - MAX_JOBS
            for jid, job in finished[:overflow]:
                _jobs.pop(jid, None)
                shutil.rmtree(job.job_dir, ignore_errors=True)


class ExternalModelBody(BaseModel):
    name: str
    url: str | None = None
    path: str | None = None
    scale: int = Field(4, ge=1, le=8)
    sha256: str | None = None
    description: str = ""


def _ensure_worker() -> None:
    """Start the single job worker if needed.

    Safe to call while holding ``_jobs_lock``: the fast path returns without
    re-entering the lock once the worker is up (avoids non-reentrant deadlock).
    """
    global _worker_started
    if _worker_started:
        return
    with _jobs_lock:
        if _worker_started:
            return
        _worker_started = True
    t = threading.Thread(target=_worker_loop, name="upscale-job-worker", daemon=True)
    t.start()


def _settings_to_config(
    settings: JobSettings,
    *,
    input_path: Path,
    output_path: Path,
    job_dir: Path,
    cancel_event: threading.Event,
    pause_event: threading.Event,
    resume_from_checkpoint: bool = False,
) -> UpscaleConfig:
    # Never honor client-supplied weights_dir from API settings (path escape).
    weights = None
    device = settings.device
    return UpscaleConfig(
        input=input_path,
        output=output_path,
        scale=settings.scale,
        target_size=settings.target_size,
        width=settings.width,
        height=settings.height,
        model=settings.model,
        algorithm=settings.algorithm,
        tile=settings.tile,
        overlap=settings.overlap,
        device=device,
        cpu=device == "cpu",
        half=settings.half,
        weights_dir=weights,
        tiff_compression=settings.tiff_compression,
        quiet=True,
        fast=settings.fast,
        sr_alpha=settings.sr_alpha,
        crop=settings.crop.as_tuple() if settings.crop else None,
        strength=settings.strength,
        denoise=settings.denoise,
        blend_model=settings.blend_model,
        blend=settings.blend,
        format=settings.format,
        quality=settings.quality,
        filters=settings.filters,
        vectorize=settings.vectorize,
        name_pattern=settings.name_pattern,
        disk_intermediate_threshold=(
            settings.disk_intermediate_threshold
            if settings.disk_intermediate_threshold is not None
            else UpscaleConfig().disk_intermediate_threshold
        ),
        job_dir=job_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        mid_pass_previews=settings.mid_pass_previews,
        cancel_event=cancel_event,
        pause_event=pause_event,
    )


def _output_suffix(settings: JobSettings) -> str:
    fmt = settings.format
    if fmt == "auto":
        return ".png"
    return {
        "jpeg": ".jpg",
        "jpeg_exif": ".jpg",
        "tiff": ".tif",
        "bmp": ".bmp",
        "tga": ".tga",
        "jp2": ".jp2",
        "psd": ".psd",
        "psb": ".psb",
        "webp": ".webp",
        "png": ".png",
    }.get(fmt, f".{fmt}")


def _push_event(job: JobRecord, event: dict[str, Any]) -> None:
    job.events.put(event)


def _active_run_seconds(job: JobRecord) -> float:
    import time

    if job.run_started_at is None:
        return 0.0
    now = time.time()
    paused = job.paused_seconds
    if job.pause_started_at is not None:
        paused += max(0.0, now - job.pause_started_at)
    return max(0.0, now - job.run_started_at - paused)


def _record_job_speed(job: JobRecord, seconds: float) -> None:
    """Learn MP/s from a finished job for future /api/estimate calls."""
    try:
        from PIL import Image

        from .estimates import output_dimensions
        from .models import registry_scale
        from .speed_stats import normalize_device, record_from_job
        from .system import gather_system_info

        with Image.open(job.input_path) as im:
            src_w, src_h = im.size
        s = job.settings
        if s.crop is not None:
            src_w = max(1, int(s.crop.w))
            src_h = max(1, int(s.crop.h))
        _ow, _oh, scale = output_dimensions(
            src_w,
            src_h,
            scale=s.scale,
            target_size=s.target_size,
            width=s.width,
            height=s.height,
        )
        sys_info = gather_system_info(device=s.device, fp16=s.half)
        device = normalize_device(sys_info.device or s.device)
        record_from_job(
            model=s.model,
            device=device,
            seconds=seconds,
            src_w=src_w,
            src_h=src_h,
            scale=float(scale),
            half=bool(s.half),
            algorithm=s.algorithm,
            model_scale=registry_scale(s.model),
        )
    except Exception:  # noqa: BLE001
        pass


def _run_job(job_id: str) -> None:
    import time

    from .tiling import clear_mask_cache

    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        # Cancelled while still queued — do not load models / start work.
        if job.cancel_event.is_set():
            job.state = JobState.cancelled
            _push_event(
                job, {"type": "cancelled", "pct": 0, "message": "cancelled"}
            )
            return
        job.state = JobState.running
        job.run_started_at = time.time()
        job.pause_started_at = None
        job.paused_seconds = 0.0
    out_path = job.job_dir / f"result{_output_suffix(job.settings)}"
    resume_ck = bool(job.resume_from_checkpoint)
    job.resume_from_checkpoint = False
    cfg = _settings_to_config(
        job.settings,
        input_path=job.input_path,
        output_path=out_path,
        job_dir=job.job_dir,
        cancel_event=job.cancel_event,
        pause_event=job.pause_event,
        resume_from_checkpoint=resume_ck,
    )
    job.cli_hint = settings_to_cli(cfg)

    def on_progress(ev: dict) -> None:
        if ev.get("pct") is not None:
            job.last_pct = float(ev["pct"])
        # Engine may emit type=done before display.png exists. The UI closes SSE
        # on the first terminal event and fetches /display — suppress here and
        # let _run_job emit done/error/cancelled after preview is ready.
        if ev.get("type") in ("done", "error", "cancelled"):
            return
        if job.pause_event.is_set() and job.state != JobState.paused:
            job.state = JobState.paused
            job.pause_started_at = time.time()
            _push_event(
                job,
                {
                    "type": "paused",
                    "pct": job.last_pct,
                    "message": "paused",
                },
            )
        _push_event(job, ev)

    try:
        # Honor pause before acquiring the GPU lock so a paused queued job
        # does not block compare / benchmark.
        while job.pause_event.is_set():
            if job.cancel_event.is_set():
                raise JobCancelled("cancelled")
            time.sleep(0.05)
        with _execute_lock:
            if job.cancel_event.is_set():
                raise JobCancelled("cancelled")
            actual = execute(cfg, on_progress=on_progress, echo=lambda _m: None)
            # Clear mask cache while still holding the lock so compare cannot
            # race make_tile_weight against OrderedDict.clear().
            clear_mask_cache()
        job.output_path = actual
        # Sidecar so /result finds the file after name_pattern renames.
        try:
            (job.job_dir / "output_name.txt").write_text(
                actual.name, encoding="utf-8"
            )
        except OSError:
            pass
        display_path = job.job_dir / "display.png"
        try:
            _make_preview(actual, display_path, max_edge=4096)
        except Exception:  # noqa: BLE001
            # Do not fall back to the full master (can be a huge TIFF/PSB).
            pass
        job.state = JobState.done
        try:
            from .checkpoint import clear_strip_checkpoint

            clear_strip_checkpoint(job.job_dir)
        except Exception:  # noqa: BLE001
            pass
        _record_job_speed(job, _active_run_seconds(job))
        _push_event(
            job,
            {
                "type": "done",
                "pct": 1.0,
                "message": "done",
                "result": str(actual),
                "display": str(display_path) if display_path.is_file() else None,
                "cli": job.cli_hint,
            },
        )
    except JobCancelled:
        job.state = JobState.cancelled
        _push_event(job, {"type": "cancelled", "pct": job.last_pct, "message": "cancelled"})
    except Exception as exc:  # noqa: BLE001
        job.state = JobState.error
        job.error = str(exc)
        _push_event(job, {"type": "error", "pct": job.last_pct, "message": str(exc)})
    finally:
        _purge_expired_jobs()


def _worker_loop() -> None:
    while True:
        job_id = _job_queue.get()
        if job_id is None:
            break
        try:
            _run_job(job_id)
        finally:
            _job_queue.task_done()


def _model_tags(name: str, meta: dict) -> list[str]:
    raw = meta.get("tags")
    if isinstance(raw, list) and raw:
        return [str(t) for t in raw]
    if meta.get("external"):
        return ["External"]
    blob = f"{name} {meta.get('description', '')}".lower()
    if any(k in blob for k in ("anime", "illustration", "video")):
        return ["Anime"]
    if any(k in blob for k in ("span", "fast", "ch48", "nomosuni_span")):
        return ["Fast"]
    return ["Photo"]


def _model_info_list(weights_dir: Path | None = None) -> list[ModelInfo]:
    from .models import prune_stale_external_models

    wdir = weights_dir or DEFAULT_WEIGHTS_DIR
    prune_stale_external_models(wdir)
    catalog = all_models(wdir)
    out: list[ModelInfo] = []
    for name, meta in sorted(catalog.items()):
        source = "external" if meta.get("external") else "built-in"
        status = model_status(name, wdir)
        url_path = (meta.get("url") or "").split("?")[0]
        ext = Path(url_path).suffix or ".pth"
        dest = wdir / f"{name}{ext}"
        size_bytes = meta.get("size_bytes")
        if size_bytes is None and dest.is_file():
            size_bytes = dest.stat().st_size
        out.append(
            ModelInfo(
                name=name,
                description=meta.get("description", ""),
                scale=int(meta.get("scale", 4)),
                source=source,  # type: ignore[arg-type]
                status=status,  # type: ignore[arg-type]
                path=str(dest) if dest.exists() else meta.get("path"),
                sha256=meta.get("sha256"),
                url=meta.get("url"),
                license=meta.get("license"),
                tags=_model_tags(name, meta),
                size_bytes=int(size_bytes) if size_bytes is not None else None,
            )
        )
    return out


def _make_preview(src: Path, dest: Path, *, max_edge: int = 512) -> None:
    from .io_utils import write_display_preview

    write_display_preview(src, dest, max_edge=max_edge)


@app.post("/api/decode-preview")
async def decode_preview(file: UploadFile = File(...)) -> Response:
    """Decode any supported input format to a browser-safe PNG preview.

    Used by the web UI for TGA / PSD / PSB / JP2 / exotic TIFF that
    ``<img>`` cannot display. Headers carry original pixel size + alpha flag.
    """
    import io
    import tempfile

    import numpy as np
    from PIL import Image

    from .io_utils import load_rgba, require_raster_suffix

    try:
        suffix = require_raster_suffix(file.filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if file.content_type:
        from .io_utils import is_vector_input

        if is_vector_input(file.filename, file.content_type):
            raise HTTPException(
                400,
                "Vector images are not supported. "
                "Rasterize to PNG, TIFF, or another supported bitmap format first.",
            )
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload exceeds limit of {MAX_UPLOAD_BYTES} bytes")
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
        # Cap preview for UI responsiveness; full file is still uploaded on run.
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


_UI_NO_CACHE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}
_LOCALE_RE = re.compile(r"^[a-z]{2}(?:-[A-Za-z]+)?$")
_UI_RESERVED = frozenset({"api", "ui", "mcp", "docs", "assets"})


def _enabled_locales() -> set[str]:
    return {str(x["code"]) for x in LANGUAGE_CATALOG if x.get("enabled")}


def _normalize_locale(code: str | None) -> str | None:
    if not code:
        return None
    raw = code.strip()
    if raw in _UI_RESERVED or not _LOCALE_RE.fullmatch(raw):
        return None
    enabled = _enabled_locales()
    if raw in enabled:
        return raw
    primary = raw.split("-", 1)[0]
    if primary in enabled:
        return primary
    return None


def _prefer_locale(request: Request) -> str:
    cookie = _normalize_locale(request.cookies.get("upscale_lang"))
    if cookie:
        return cookie
    for part in (request.headers.get("accept-language") or "").split(","):
        tag = part.split(";", 1)[0].strip().lower().replace("_", "-")
        if not tag or tag == "*":
            continue
        hit = _normalize_locale(tag) or _normalize_locale(tag.split("-", 1)[0])
        if hit:
            return hit
    return "en"


def _prefer_view(request: Request) -> str:
    """Last Simple/Advanced choice from cookie; default Simple."""
    raw = (request.cookies.get("upscale_view") or "").strip().lower()
    if raw in ("simple", "advanced"):
        return raw
    return "simple"


def _ui_page_file(page: str) -> Path | None:
    path = (UI_DIR / f"{page}.html").resolve()
    if not path.is_file() or UI_DIR.resolve() not in path.parents:
        return None
    return path


def _serve_ui_landing(request: Request) -> Response:
    """Landing restores last view (cookie); default Simple. Advanced is /{locale}/index.html."""
    order = ("index", "simple") if _prefer_view(request) == "advanced" else ("simple", "index")
    for name in order:
        path = _ui_page_file(name)
        if path is not None:
            return FileResponse(path, headers=_UI_NO_CACHE)
    return RedirectResponse("/api/system")


@app.get("/", response_model=None)
def root(request: Request) -> Response:
    """Redirect bare root into a locale-prefixed UI path."""
    return RedirectResponse(f"/{_prefer_locale(request)}/", status_code=307)


@app.get("/{locale}/", response_model=None)
def locale_root(locale: str, request: Request) -> Response:
    loc = _normalize_locale(locale)
    if not loc:
        raise HTTPException(404, "Locale not found")
    return _serve_ui_landing(request)


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


@app.post("/api/estimate")
def api_estimate(body: EstimateBody) -> dict[str, Any]:
    """Honest time range; prefers learned MP/s from past jobs on this machine."""
    from .speed_stats import lookup_measured_mp_s, normalize_device, work_megapixels

    name = body.model
    ready = body.model_ready
    if ready is None:
        ready = model_status(name) == "installed"
    meta = all_models().get(name) or MODELS.get(name) or {}
    model_bytes = meta.get("size_bytes")
    # Prefer on-disk size when weights are installed
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


@app.get("/{locale}/{page}.html", include_in_schema=False)
def locale_ui_page(locale: str, page: str) -> FileResponse:
    loc = _normalize_locale(locale)
    if not loc:
        raise HTTPException(404, "Locale not found")
    path = _ui_page_file(page)
    if path is None:
        raise HTTPException(404, "Not found")
    return FileResponse(path, headers=_UI_NO_CACHE)


@app.get("/{page}.html", include_in_schema=False)
def ui_page_legacy(page: str, request: Request) -> RedirectResponse:
    """Old unlocalized URLs redirect into /{locale}/{page}.html."""
    if _ui_page_file(page) is None:
        raise HTTPException(404, "Not found")
    return RedirectResponse(
        f"/{_prefer_locale(request)}/{page}.html",
        status_code=307,
    )


@app.get("/ui/{asset_path:path}", include_in_schema=False)
def ui_asset(asset_path: str) -> FileResponse:
    path = (UI_DIR / asset_path).resolve()
    if not path.is_file() or UI_DIR.resolve() not in path.parents:
        raise HTTPException(404, "Not found")
    return FileResponse(path, headers=_UI_NO_CACHE)




def _get_job(job_id: str) -> JobRecord:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


def _safe_job_dir(job_id: str) -> Path:
    """Resolve job dir; reject path traversal."""
    if (
        not job_id
        or ".." in job_id
        or "/" in job_id
        or "\\" in job_id
        or job_id.startswith("compare-")
    ):
        raise HTTPException(400, "Invalid job id")
    return JOBS_ROOT / job_id


def _find_result_file(job_dir: Path) -> Path | None:
    sidecar = job_dir / "output_name.txt"
    if sidecar.is_file():
        try:
            name = sidecar.read_text(encoding="utf-8").strip()
            if name and ".." not in name and "/" not in name and "\\" not in name:
                p = job_dir / name
                if p.is_file() and p.stat().st_size > 0:
                    return p
        except OSError:
            pass
    preferred = (
        "result.png",
        "result.jpg",
        "result.jpeg",
        "result.webp",
        "result.tif",
        "result.tiff",
        "result.bmp",
        "result.tga",
        "result.jp2",
        "result.psd",
        "result.psb",
    )
    for name in preferred:
        p = job_dir / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    matches = sorted(
        p for p in job_dir.glob("result.*") if p.is_file() and p.stat().st_size > 0
    )
    return matches[0] if matches else None


def _rehydrate_job_from_disk(job_id: str) -> JobRecord:
    """Rebuild an in-memory JobRecord after process restart (settings.json + input.*)."""
    job_dir = _safe_job_dir(job_id)
    settings_path = job_dir / "settings.json"
    if not settings_path.is_file():
        raise HTTPException(404, "Job not found (no settings on disk)")
    try:
        js = JobSettings.model_validate_json(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"Corrupt settings.json: {exc}") from exc
    inputs = sorted(job_dir.glob("input.*"))
    if not inputs:
        raise HTTPException(400, "Input file missing for this job")
    job = JobRecord(
        id=job_id,
        job_dir=job_dir,
        input_path=inputs[0],
        settings=js,
        state=JobState.cancelled,
    )
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def _png_readable(path: Path) -> bool:
    """True if Pillow can fully decode the image (rejects truncated/corrupt PNGs)."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
        return True
    except Exception:  # noqa: BLE001
        return False


def _discard_bad_checkpoint(job_dir: Path) -> None:
    try:
        from .checkpoint import clear_strip_checkpoint

        clear_strip_checkpoint(job_dir)
    except Exception:  # noqa: BLE001
        pass




def _strip_checkpoint_info_from_dir(job_dir: Path) -> dict[str, Any] | None:
    """Return resume metadata only when the strip checkpoint is actually usable."""
    from .checkpoint import load_strip_checkpoint

    loaded = load_strip_checkpoint(job_dir)
    if loaded is None:
        return None
    meta, _extras = loaded
    # Zero flush means no finished bands — nothing useful to continue from.
    if meta.flush_row <= 0 or meta.out_h <= 0 or meta.out_w <= 0:
        return None
    return {
        "pass_index": meta.pass_index,
        "flush_row": meta.flush_row,
        "out_h": meta.out_h,
        "out_w": meta.out_w,
        "pct_approx": round(meta.flush_row / max(meta.out_h, 1), 3),
    }


def _strip_checkpoint_info(job: JobRecord) -> dict[str, Any] | None:
    return _strip_checkpoint_info_from_dir(job.job_dir)




















































_compare_runs: dict[str, Path] = {}
_compare_jobs: dict[str, Any] = {}
_compare_lock = threading.Lock()













from upscaler.api import include_routers

include_routers(app)

def _register_mcp_route() -> None:
    try:
        from .mcp_server import mount_mcp_on_fastapi

        mount_mcp_on_fastapi(app)
    except ImportError:
        pass


_register_mcp_route()


def run_server(host: str = "127.0.0.1", port: int = 8787) -> None:
    import ipaddress
    import os
    import signal

    import uvicorn

    # Refuse non-loopback binds unless explicitly opted in (unauthenticated API).
    allow_remote = os.environ.get("UPSCALE_ALLOW_REMOTE", "").strip() in (
        "1",
        "true",
        "yes",
    )
    loopback = False
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host in ("localhost",)
    if not loopback and not allow_remote:
        raise RuntimeError(
            f"Refusing to bind non-loopback host {host!r}. "
            "Set UPSCALE_ALLOW_REMOTE=1 to allow (API is unauthenticated)."
        )

    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    # Drop stale job dirs left from a previous process (keep valid strip CKs).
    try:
        _purge_stale_job_dirs_on_disk()
    except Exception:  # noqa: BLE001
        pass

    # Windows consoles often leave uvicorn stuck after Ctrl+C (CUDA/native
    # threads). Force-exit so web.bat can be restarted without stop-web.bat.
    def _force_exit(_signum=None, _frame=None) -> None:
        os._exit(0)

    signal.signal(signal.SIGINT, _force_exit)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _force_exit)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_graceful_shutdown=1,
    )
