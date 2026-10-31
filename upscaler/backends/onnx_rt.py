"""ONNX Runtime upscale path — no PyTorch at runtime."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import UpscaleConfig
from ..job_prep import clamp_overlap, finalize_result, prepare_job
from ..models import FAST_MODEL, DEFAULT_WEIGHTS_DIR, registry_scale
from ..pipeline import compute_passes, run
from ..tiling import JobCancelled

# Optional remote URL for pre-exported ONNX (publish when CI exports).
ONNX_MODELS: dict[str, dict] = {
    "4x-spanx4-ch48": {
        "scale": 4,
        "filename": "4x-spanx4-ch48.onnx",
        "description": "SPAN 4x (ONNX). Preferred slim desktop model.",
    },
    "RealESRGAN_x4plus": {
        "scale": 4,
        "filename": "RealESRGAN_x4plus.onnx",
        "description": "RealESRGAN x4plus (ONNX).",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "scale": 4,
        "filename": "RealESRGAN_x4plus_anime_6B.onnx",
        "description": "Anime 6B (ONNX).",
    },
    "realesr-general-x4v3": {
        "scale": 4,
        "filename": "realesr-general-x4v3.onnx",
        "description": "General x4v3 (ONNX).",
    },
    "realesr-general-wdn-x4v3": {
        "scale": 4,
        "filename": "realesr-general-wdn-x4v3.onnx",
        "description": "General WDN x4v3 (ONNX).",
    },
}


def _weights_dir(cfg: UpscaleConfig | None = None) -> Path:
    env = os.environ.get("UPSCALE_MODELS_DIR")
    if env:
        return Path(env)
    if cfg and cfg.weights_dir:
        return Path(cfg.weights_dir)
    return DEFAULT_WEIGHTS_DIR


def onnx_path_for(name: str, weights_dir: Path) -> Path:
    meta = ONNX_MODELS.get(name)
    filename = meta["filename"] if meta else f"{name}.onnx"
    return Path(weights_dir) / filename


def ensure_onnx_model(name: str, weights_dir: Path | None = None) -> Path:
    """Return path to .onnx weights. Does not download yet — must be exported/placed."""
    wdir = weights_dir or DEFAULT_WEIGHTS_DIR
    wdir = Path(wdir)
    wdir.mkdir(parents=True, exist_ok=True)
    path = onnx_path_for(name, wdir)
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"ONNX model not found: {path}\n"
        f"Export on a Torch machine: py -3 scripts/export_onnx.py {name} -o {path}"
    )


def _ort_session(model_path: Path):
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    preferred = []
    for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"):
        if p in available:
            preferred.append(p)
    if not preferred:
        preferred = ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), opts, providers=preferred), preferred[0]


def make_onnx_infer_fn(session, input_name: str) -> Callable[[np.ndarray], np.ndarray]:
    from ..infer_util import nchw_float_to_uint8, patch_to_nchw_float

    def infer(patch: np.ndarray) -> np.ndarray:
        x = patch_to_nchw_float(patch)
        out = session.run(None, {input_name: x})[0]
        return nchw_float_to_uint8(out)

    return infer


def execute_onnx(
    cfg: UpscaleConfig,
    *,
    progress_callback: Callable[[str, float], None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
    echo: Callable[[str], None] | None = None,
) -> Path:
    prep = prepare_job(cfg, on_progress=on_progress, echo=echo)
    model_name = FAST_MODEL if cfg.fast else cfg.model
    wdir = _weights_dir(cfg)

    model_scale = int(
        ONNX_MODELS.get(model_name, {}).get("scale") or registry_scale(model_name)
    )
    if cfg.tile in ("auto", None):
        # Rough RAM-based tile (ORT has no cuda.mem_get_info here).
        try:
            import psutil

            avail = int(psutil.virtual_memory().available * 0.25)
        except Exception:  # noqa: BLE001
            avail = 512 * 1024 * 1024
        bytes_per_px = 8000
        tile_guess = int(max(128, min(512, (avail / bytes_per_px) ** 0.5)))
        tile_int = (tile_guess // 64) * 64 or 256
    else:
        tile_int = int(cfg.tile)
    overlap = clamp_overlap(cfg.overlap, tile_int)

    infer = None
    provider = "cpu"
    if prep.scale > 1.0:
        path = ensure_onnx_model(model_name, wdir)
        if prep.show:
            prep.log(f"ONNX model: {path.name}")
        session, provider = _ort_session(path)
        input_name = session.get_inputs()[0].name
        infer = make_onnx_infer_fn(session, input_name)

    passes = compute_passes(prep.scale, model_scale) if prep.scale > 1 else 0
    if prep.show:
        prep.log(
            f"Backend: onnx/{provider}  |  Model: {model_name}  |  "
            f"Scale: {prep.scale:.3f}x  |  Passes: {passes}  |  Tile: {tile_int}"
        )

    if cfg.blend_model:
        warnings.warn("blend_model is not supported on the ONNX backend yet; ignored.")
    if cfg.denoise is not None and abs(cfg.denoise - 0.5) > 1e-6:
        warnings.warn("--denoise DNI is not supported on the ONNX backend yet; ignored.")

    run_tw = round(prep.img.shape[1] * prep.scale)
    run_th = round(prep.img.shape[0] * prep.scale)

    try:
        result = run(
            img=prep.img,
            scale=prep.scale,
            descriptor=None,
            device=type("D", (), {"type": "cpu"})(),
            fp16=False,
            tile=tile_int,
            overlap=overlap,
            show_progress=prep.show,
            model_scale=model_scale,
            disk_intermediate_threshold=cfg.disk_intermediate_threshold,
            progress_callback=progress_callback,
            cancel_event=cfg.cancel_event,
            pause_event=cfg.pause_event,
            strength=1.0,
            job_dir=cfg.job_dir,
            resume_from_checkpoint=bool(getattr(cfg, "resume_from_checkpoint", False)),
            mid_pass_previews=cfg.mid_pass_previews and cfg.job_dir is not None,
            on_progress=on_progress,
            target_w=run_tw,
            target_h=run_th,
            infer_fn=infer,
        )
    except JobCancelled:
        if on_progress:
            on_progress({"type": "cancelled", "pct": 0, "message": "cancelled"})
        raise

    # Alpha: Lanczos only on slim path
    out_alpha = prep.alpha
    return finalize_result(
        prep,
        result,
        on_progress=on_progress,
        model_label=model_name,
        out_alpha=out_alpha,
    )
