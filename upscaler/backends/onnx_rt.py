"""ONNX Runtime upscale path — no PyTorch at runtime."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import UpscaleConfig
from ..job_prep import clamp_overlap, finalize_result, prepare_job
from ..models import DEFAULT_WEIGHTS_DIR, FAST_MODEL, registry_scale
from ..pipeline import compute_passes, run
from ..tiling import JobCancelled

# Optional remote URL for pre-exported ONNX (publish when CI exports).
# sha256/url filled when a release host is published; until then ensure_onnx_model
# looks on disk and raises a clear first-run message.
ONNX_MODELS: dict[str, dict] = {
    "4x-spanx4-ch48": {
        "scale": 4,
        "filename": "4x-spanx4-ch48.onnx",
        "description": "SPAN 4x (ONNX). Preferred slim desktop model.",
        "sha256": None,
        "url": None,
    },
    "RealESRGAN_x4plus": {
        "scale": 4,
        "filename": "RealESRGAN_x4plus.onnx",
        "description": "RealESRGAN x4plus (ONNX).",
        "sha256": None,
        "url": None,
    },
    "RealESRGAN_x4plus_anime_6B": {
        "scale": 4,
        "filename": "RealESRGAN_x4plus_anime_6B.onnx",
        "description": "Anime 6B (ONNX).",
        "sha256": None,
        "url": None,
    },
    "realesr-general-x4v3": {
        "scale": 4,
        "filename": "realesr-general-x4v3.onnx",
        "description": "General x4v3 (ONNX).",
        "sha256": None,
        "url": None,
    },
    "realesr-general-wdn-x4v3": {
        "scale": 4,
        "filename": "realesr-general-wdn-x4v3.onnx",
        "description": "General WDN x4v3 (ONNX).",
        "sha256": None,
        "url": None,
    },
}

# Cache ORT sessions by (path, providers) for the process lifetime.
_ORT_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], object] = {}


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
    """Return path to .onnx weights; download when url+sha256 are configured."""
    wdir = weights_dir or DEFAULT_WEIGHTS_DIR
    wdir = Path(wdir)
    wdir.mkdir(parents=True, exist_ok=True)
    path = onnx_path_for(name, wdir)
    meta = ONNX_MODELS.get(name, {})
    if path.is_file():
        expected = meta.get("sha256")
        if expected:
            from ..models import _sha256

            if _sha256(path) != expected:
                path.unlink(missing_ok=True)
            else:
                return path
        else:
            return path
    url = meta.get("url")
    sha = meta.get("sha256")
    if url and sha:
        from ..models import _download_with_retries, _sha256

        _download_with_retries(url, path)
        if _sha256(path) != sha:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"ONNX hash mismatch for {name}")
        return path
    raise FileNotFoundError(
        f"ONNX model not found: {path}\n"
        "First-run: export on a Torch machine with\n"
        f"  py -3 scripts/export_onnx.py {name} -o {path}\n"
        "or place a pre-exported .onnx in the models folder."
    )


def _ort_session(model_path: Path):
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    # Prefer CUDA when present; else DirectML (Windows slim); else CPU.
    preferred = []
    for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"):
        if p in available:
            preferred.append(p)
    if not preferred:
        preferred = ["CPUExecutionProvider"]
    key = (str(model_path.resolve()), tuple(preferred))
    cached = _ORT_SESSION_CACHE.get(key)
    if cached is not None:
        return cached, preferred[0]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), opts, providers=preferred)
    _ORT_SESSION_CACHE[key] = session
    return session, preferred[0]


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
        warnings.warn(
            "blend_model weight interpolation needs Torch; ignored on ONNX. "
            "Export a pre-blended ONNX or use the Torch backend.",
            stacklevel=2,
        )
    if cfg.denoise is not None and abs(cfg.denoise - 0.5) > 1e-6:
        # Soft parity: if the WDN twin is on disk, switch to it when denoise>0.5.
        wdn = "realesr-general-wdn-x4v3"
        general = "realesr-general-x4v3"
        if model_name in (general, wdn) and cfg.denoise > 0.5:
            alt = onnx_path_for(wdn, wdir)
            if alt.is_file() and model_name != wdn:
                model_name = wdn
                path = alt
                session, provider = _ort_session(path)
                input_name = session.get_inputs()[0].name
                infer = make_onnx_infer_fn(session, input_name)
                if prep.show:
                    prep.log(f"ONNX denoise>0.5 → switched to {wdn}")
            else:
                warnings.warn(
                    "--denoise DNI is not supported on ONNX (no WDN twin on disk); ignored.",
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "--denoise DNI is not supported on the ONNX backend yet; ignored.",
                stacklevel=2,
            )

    from ..job_prep import scaled_size

    run_tw, run_th = scaled_size(prep.img.shape[1], prep.img.shape[0], prep.scale)

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
