"""MCP stdio server exposing upscale tools for AI agents."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from .compare import compare_models as run_model_compare
from .config import UpscaleConfig
from .engine import execute
from .help_content import get_help as load_help
from .help_content import list_topics
from .models import DEFAULT_WEIGHTS_DIR, add_external_model, all_models, model_status
from .schema import CropRect
from .system import gather_system_info

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover — optional extra
    FastMCP = None  # type: ignore[misc, assignment]

_mcp_jobs_lock = threading.Lock()
_mcp_jobs: dict[str, threading.Event] = {}


def _register_job() -> tuple[str, threading.Event]:
    job_id = uuid.uuid4().hex
    ev = threading.Event()
    with _mcp_jobs_lock:
        _mcp_jobs[job_id] = ev
    return job_id, ev


def _clear_job(job_id: str) -> None:
    with _mcp_jobs_lock:
        _mcp_jobs.pop(job_id, None)


def _cfg(
    input_path: str,
    output_path: str,
    *,
    scale: float | None = 4.0,
    model: str = "RealESRGAN_x4plus",
    device: str = "auto",
    half: bool = False,
    cpu: bool = False,
    cancel_event: threading.Event | None = None,
    **extra: Any,
) -> UpscaleConfig:
    return UpscaleConfig(
        input=Path(input_path),
        output=Path(output_path),
        scale=scale,
        model=model,
        device=device,
        half=half,
        cpu=cpu or device == "cpu",
        quiet=True,
        cancel_event=cancel_event,
        **extra,
    )


def build_mcp() -> Any:
    """Create and configure the FastMCP application (requires ``upscaler[mcp]``)."""
    if FastMCP is None:
        raise ImportError("Install MCP support: pip install 'upscaler[mcp]'")

    mcp = FastMCP(
        "upscale",
        instructions=(
            "Tools to upscale images with Real-ESRGAN / spandrel models, "
            "batch folders, compare models, and read bundled help topics."
        ),
    )

    @mcp.tool()
    def upscale_image(
        input_path: str,
        output_path: str,
        scale: float = 4.0,
        model: str = "RealESRGAN_x4plus",
        device: str = "auto",
        half: bool = False,
        target_size: int | None = None,
    ) -> dict[str, Any]:
        """Upscale a single image file to OUTPUT_PATH."""
        job_id, cancel = _register_job()
        try:
            cfg = _cfg(
                input_path,
                output_path,
                scale=scale if target_size is None else None,
                target_size=target_size,
                model=model,
                device=device,
                half=half,
                cancel_event=cancel,
            )
            out = execute(cfg)
            return {"job_id": job_id, "status": "done", "output": str(out)}
        except Exception as exc:  # noqa: BLE001
            return {"job_id": job_id, "status": "error", "error": str(exc)}
        finally:
            _clear_job(job_id)

    @mcp.tool()
    def batch_upscale(
        input_dir: str,
        output_dir: str,
        scale: float = 4.0,
        model: str = "RealESRGAN_x4plus",
        device: str = "auto",
        half: bool = False,
    ) -> dict[str, Any]:
        """Upscale all images in INPUT_DIR into OUTPUT_DIR (non-recursive)."""
        from .cli import IMAGE_GLOBS

        ind = Path(input_dir)
        outd = Path(output_dir)
        outd.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for pattern in IMAGE_GLOBS:
            files.extend(sorted(ind.glob(pattern)))
            files.extend(sorted(ind.glob(pattern.upper())))
        seen: set[Path] = set()
        unique = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        if not unique:
            return {"status": "error", "error": f"No images in {input_dir}"}

        job_id, cancel = _register_job()
        results: list[dict[str, Any]] = []
        try:
            for src in unique:
                if cancel.is_set():
                    return {"job_id": job_id, "status": "cancelled", "results": results}
                dest = outd / src.name
                cfg = _cfg(
                    str(src),
                    str(dest),
                    scale=scale,
                    model=model,
                    device=device,
                    half=half,
                    cancel_event=cancel,
                )
                try:
                    path = execute(cfg)
                    results.append({"input": str(src), "output": str(path), "ok": True})
                except Exception as exc:  # noqa: BLE001
                    results.append({"input": str(src), "ok": False, "error": str(exc)})
            return {"job_id": job_id, "status": "done", "results": results}
        finally:
            _clear_job(job_id)

    @mcp.tool(name="compare_models")
    def compare_models_mcp(
        input_path: str,
        output_dir: str,
        crop: str,
        models: str,
        scale: float = 4.0,
        device: str = "auto",
        half: bool = False,
    ) -> dict[str, Any]:
        """Compare comma-separated MODELS on crop region ``x,y,w,h``."""
        parts = [int(x.strip()) for x in crop.split(",")]
        if len(parts) != 4:
            return {"status": "error", "error": "crop must be x,y,w,h"}
        names = [m.strip() for m in models.split(",") if m.strip()]
        rect = CropRect(x=parts[0], y=parts[1], w=parts[2], h=parts[3])
        rows = run_model_compare(
            Path(input_path),
            Path(output_dir),
            models=names,
            crop=rect,
            scale=scale,
            device=device,
            half=half,
        )
        return {"status": "done", "results": rows}

    @mcp.tool()
    def list_models(weights_dir: str | None = None) -> dict[str, Any]:
        """List built-in and external models with download status."""
        wdir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
        catalog = all_models(wdir)
        items = []
        for name, meta in sorted(catalog.items()):
            items.append(
                {
                    "name": name,
                    "status": model_status(name, wdir),
                    "scale": meta.get("scale", 4),
                    "description": meta.get("description", ""),
                }
            )
        return {"models": items}

    @mcp.tool()
    def add_model(
        name: str,
        url_or_path: str,
        scale: int = 4,
        sha256: str | None = None,
        weights_dir: str | None = None,
    ) -> dict[str, Any]:
        """Register an external model from URL or local path."""
        from .models import validate_model_name

        validate_model_name(name)
        wdir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
        p = Path(url_or_path)
        if p.exists():
            add_external_model(name, path=str(p), scale=scale, sha256=sha256, weights_dir=wdir)
        else:
            add_external_model(name, url=url_or_path, scale=scale, sha256=sha256, weights_dir=wdir)
        return {"name": name, "status": model_status(name, wdir)}

    @mcp.tool()
    def system_info(device: str = "auto", half: bool = False) -> dict[str, Any]:
        """Report device, VRAM, RAM, and recommended tile size."""
        info = gather_system_info(device=device, fp16=half)
        return info.model_dump()

    @mcp.tool()
    def get_help(topic: str = "mcp", lang: str = "en") -> dict[str, str]:
        """Return markdown help for a topic (see list in /api/help)."""
        body = load_help(topic, lang)
        return {"topic": topic, "lang": lang, "markdown": body}

    @mcp.tool()
    def cancel_job(job_id: str) -> dict[str, str]:
        """Signal cancellation for a running MCP upscale/batch job."""
        with _mcp_jobs_lock:
            ev = _mcp_jobs.get(job_id)
        if ev is None:
            return {"status": "not_found", "job_id": job_id}
        ev.set()
        return {"status": "cancelling", "job_id": job_id}

    return mcp


def mcp_tool_catalog() -> dict[str, Any]:
    """JSON catalog for HTTP ``GET /mcp`` (no MCP session required)."""
    topics = list_topics("en")
    tools = [
        "upscale_image",
        "batch_upscale",
        "compare_models",
        "list_models",
        "add_model",
        "system_info",
        "get_help",
        "cancel_job",
    ]
    return {
        "transport_stdio": "upscale mcp",
        "tools": tools,
        "help_topics": topics,
        "note": "Streamable HTTP MCP can be enabled via FastMCP; stdio is the default agent transport.",
    }


def run_stdio() -> None:
    """Entry point for ``upscale mcp``."""
    build_mcp().run(transport="stdio")


def mount_mcp_on_fastapi(app: Any) -> None:
    """Attach ``GET /mcp`` info route to an existing FastAPI app."""
    from fastapi import FastAPI

    if not isinstance(app, FastAPI):
        return

    @app.get("/mcp")
    def _mcp_note() -> dict[str, Any]:
        return mcp_tool_catalog()
