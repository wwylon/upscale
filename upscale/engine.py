"""Shared upscale execution path for CLI, batch, and web UI."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import click
import numpy as np

from .backends import resolve_backend
from .config import UpscaleConfig
from .job_prep import clamp_overlap, finalize_result, prepare_job
from .models import (
    FAST_MODEL,
    auto_tile_size,
    ensure_model,
    load_model,
    registry_scale,
    resolve_device,
)
from .pipeline import compute_passes, resolve_model_scale, run, upscale_alpha
from .tiling import JobCancelled


def _execute_classical(
    cfg: UpscaleConfig,
    *,
    progress_callback: Callable[[str, float], None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
    echo: Callable[[str], None] | None = None,
) -> Path:
    """Torch-free classical / plugin resize path."""
    from .algorithms import resize_with_algorithm
    from .algorithms.classical import backend_name
    from .tiling import JobCancelled as JC

    prep = prepare_job(cfg, on_progress=on_progress, echo=echo)
    algorithm = str(cfg.algorithm)
    if prep.show:
        prep.log(
            f"Classical algorithm: {algorithm} ({backend_name()}) → "
            f"{prep.target_w}×{prep.target_h}"
        )
    if on_progress:
        on_progress(
            {"type": "progress", "pct": 0.1, "message": f"Classical {algorithm}…"}
        )
    if algorithm == "vectorize_scale":
        h0, w0 = prep.img.shape[:2]
        if max(h0, w0) > 512:
            msg = (
                "Vectorize & scale works best on logos / flat graphics — "
                "large photos may look posterized and take longer."
            )
            if prep.show:
                prep.log(msg)
            if on_progress:
                on_progress({"type": "progress", "pct": 0.12, "message": msg})
    if cfg.cancel_event is not None and cfg.cancel_event.is_set():
        raise JC("cancelled")

    # Resize padded (or not) canvas; finalize unwraps + exact size.
    from .job_prep import scaled_size

    tw, th = scaled_size(prep.img.shape[1], prep.img.shape[0], prep.scale)
    if algorithm == "vectorize_scale":
        from .algorithms.plugins.vectorize_scale import vectorize_scale_resize
        from .schema import VectorizeSettings

        vz = getattr(cfg, "vectorize", None) or VectorizeSettings()
        result = vectorize_scale_resize(
            prep.img,
            tw,
            th,
            engine=vz.engine,
            color_precision=vz.color_precision,
            filter_speckle=vz.filter_speckle,
            corner_threshold=vz.corner_threshold,
            mode=vz.mode,
            max_trace_edge=vz.max_trace_edge,
        )
    else:
        result = resize_with_algorithm(algorithm, prep.img, tw, th)

    # Keep alpha at the padded geometry; finalize_result unwraps + resizes.
    out_alpha = prep.alpha

    path = finalize_result(
        prep,
        result,
        on_progress=on_progress,
        model_label=algorithm,
        out_alpha=out_alpha,
    )
    if progress_callback:
        progress_callback("done", 1.0)
    return path


def resolve_tile(tile: str | int, device, half: bool) -> int:
    if tile == "auto" or tile is None:
        return auto_tile_size(device, half)
    try:
        return int(tile)
    except (TypeError, ValueError) as exc:
        raise click.UsageError(
            f"Invalid --tile value: {tile!r}. Use 'auto' or an integer."
        ) from exc


def blend_parameters(dst_model, src_model, alpha: float) -> None:
    """Atomically blend ``src`` into ``dst``: dst ← (1-α)·dst + α·src.

    Builds a full replacement state dict and assigns it only after every
    parameter name and shape matches. Raises on mismatch (caller may warn).
    """
    import torch

    a = float(np.clip(alpha, 0.0, 1.0))
    dst_sd = {k: v.detach() for k, v in dst_model.state_dict().items()}
    src_sd = {k: v.detach() for k, v in src_model.state_dict().items()}
    if set(dst_sd) != set(src_sd):
        raise ValueError("blend_parameters: state dict keys differ")
    blended = {}
    for k, v_dst in dst_sd.items():
        v_src = src_sd[k]
        if v_dst.shape != v_src.shape:
            raise ValueError(f"blend_parameters: shape mismatch on {k}")
        if not torch.is_floating_point(v_dst):
            blended[k] = v_dst.clone()
        else:
            blended[k] = v_dst.float().mul(1.0 - a).add_(v_src.float(), alpha=a).to(
                dtype=v_dst.dtype
            )
    dst_model.load_state_dict(blended, strict=True)


def _apply_denoise_dni(descriptor, cfg: UpscaleConfig, model_name: str, device, half, log, show) -> None:
    """Blend general / WDN weights when --denoise is set for the general pair.

    ``denoise=0.5`` (the config default) is a sentinel meaning "leave the base
    weights alone" — not a 50/50 mix. Endpoints: 0 → general, 1 → full WDN.
    """
    import torch

    dni = cfg.denoise
    if dni is None or abs(dni - 0.5) < 1e-6:
        return
    general = "realesr-general-x4v3"
    wdn = "realesr-general-wdn-x4v3"
    if model_name not in (general, wdn):
        warnings.warn(
            f"--denoise is only applied for {general}/{wdn}; ignored for {model_name}.",
            stacklevel=2,
        )
        return
    try:
        other_name = wdn if model_name == general else general
        # denoise=1 → full WDN, denoise=0 → full general
        w_wdn = float(np.clip(dni, 0.0, 1.0))
        if model_name == wdn:
            other_path = ensure_model(other_name, cfg.resolved_weights_dir())
            other = load_model(other_path, device, half)
            a = 1.0 - w_wdn  # weight of other (general)
        else:
            other_path = ensure_model(wdn, cfg.resolved_weights_dir())
            other = load_model(other_path, device, half)
            a = w_wdn  # weight of WDN into general
        # Blend on CPU tensors when possible to avoid two full GPU residents.
        other.model.to("cpu")
        descriptor.model.to("cpu")
        blend_parameters(descriptor.model, other.model, a)
        descriptor.model.to(device)
        if show:
            log(f"Denoise DNI: {model_name} blended with {other_name} @ {dni:.2f}")
        del other
        import gc

        gc.collect()
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Denoise DNI skipped: {exc}", stacklevel=2)
        try:
            descriptor.model.to(device)
        except Exception:  # noqa: BLE001
            pass


def execute(
    cfg: UpscaleConfig,
    *,
    progress_callback: Callable[[str, float], None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
    echo: Callable[[str], None] | None = None,
) -> Path:
    """Run a single upscale from cfg.input → cfg.output. Returns path written."""
    if getattr(cfg, "algorithm", None):
        return _execute_classical(
            cfg, progress_callback=progress_callback, on_progress=on_progress, echo=echo
        )

    backend = resolve_backend()
    if backend == "onnx":
        from .backends.onnx_rt import execute_onnx

        return execute_onnx(
            cfg,
            progress_callback=progress_callback,
            on_progress=on_progress,
            echo=echo,
        )

    import torch

    prep = prepare_job(cfg, on_progress=on_progress, echo=echo)
    model_name = FAST_MODEL if cfg.fast else cfg.model
    if cfg.fast and prep.show:
        prep.log(f"Fast mode: using model {model_name}")

    dev = resolve_device(cfg.resolved_device_str())
    half = cfg.half
    if half and getattr(dev, "type", None) == "cpu":
        warnings.warn("--half is ignored on CPU (fp16 not accelerated there).", stacklevel=2)
        half = False

    tile_int = resolve_tile(cfg.tile, dev, half)
    if prep.show and cfg.tile == "auto":
        prep.log(f"Auto tile size: {tile_int}")
    overlap = clamp_overlap(cfg.overlap, tile_int)
    if overlap != cfg.overlap and prep.show:
        prep.log(f"Overlap clamped to {overlap} (must be < tile/2).")

    model_scale = registry_scale(model_name)
    descriptor = None
    if prep.scale > 1.0:
        if cfg.cancel_event is not None and cfg.cancel_event.is_set():
            raise JobCancelled("cancelled")
        if prep.show:
            prep.log("Loading model ...")
        if on_progress:
            on_progress(
                {"type": "loading_model", "pct": 0.02, "message": "Loading model…"}
            )
        model_path = ensure_model(model_name, cfg.resolved_weights_dir())
        if cfg.cancel_event is not None and cfg.cancel_event.is_set():
            raise JobCancelled("cancelled")
        descriptor = load_model(model_path, dev, half)
        if cfg.cancel_event is not None and cfg.cancel_event.is_set():
            raise JobCancelled("cancelled")
        model_scale = resolve_model_scale(descriptor, fallback=model_scale)
        _apply_denoise_dni(descriptor, cfg, model_name, dev, half, prep.log, prep.show)

        if cfg.blend_model and 0.0 < cfg.blend < 1.0:
            try:
                other_path = ensure_model(cfg.blend_model, cfg.resolved_weights_dir())
                other = load_model(other_path, dev, half)
                other.model.to("cpu")
                descriptor.model.to("cpu")
                blend_parameters(descriptor.model, other.model, float(cfg.blend))
                descriptor.model.to(dev)
                if prep.show:
                    prep.log(f"Blended with {cfg.blend_model} @ {cfg.blend:.2f}")
                del other
                import gc

                gc.collect()
                if getattr(dev, "type", None) == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Model blend skipped: {exc}", stacklevel=2)
                try:
                    descriptor.model.to(dev)
                except Exception:  # noqa: BLE001
                    pass

    passes = compute_passes(prep.scale, model_scale)
    if prep.show:
        prep.log(
            f"Device: {dev}  |  Model: {model_name}  |  "
            f"Scale: {prep.scale:.3f}x  |  Passes: {passes}  |  Model scale: {model_scale}x"
        )

    # Targets for run include seamless pad so unwrap can peel correctly.
    from .job_prep import scaled_size

    run_tw, run_th = scaled_size(prep.img.shape[1], prep.img.shape[0], prep.scale)

    try:
        result = run(
            img=prep.img,
            scale=prep.scale,
            descriptor=descriptor,
            device=dev,
            fp16=half,
            tile=tile_int,
            overlap=overlap,
            show_progress=prep.show,
            model_scale=model_scale,
            disk_intermediate_threshold=cfg.disk_intermediate_threshold,
            progress_callback=progress_callback,
            cancel_event=cfg.cancel_event,
            pause_event=cfg.pause_event,
            strength=1.0,  # strength applied in finalize against unpadded source
            job_dir=cfg.job_dir,
            resume_from_checkpoint=bool(getattr(cfg, "resume_from_checkpoint", False)),
            mid_pass_previews=cfg.mid_pass_previews and cfg.job_dir is not None,
            on_progress=on_progress,
            target_w=run_tw,
            target_h=run_th,
        )
    except JobCancelled:
        if on_progress:
            on_progress({"type": "cancelled", "pct": 0, "message": "cancelled"})
        raise

    out_alpha = prep.alpha
    if prep.alpha is not None and prep.original_mode in ("RGBA", "LA") and prep.scale != 1.0:
        if cfg.sr_alpha and descriptor is not None and prep.scale > 1.0:
            if prep.show:
                prep.log("Upscaling alpha via SR ...")
            # Target the padded canvas; finalize unwraps with the same margins as RGB.
            out_alpha = upscale_alpha(
                prep.alpha,
                scale=prep.scale,
                descriptor=descriptor,
                device=dev,
                fp16=half,
                tile=tile_int,
                overlap=overlap,
                show_progress=False,
                model_scale=model_scale,
                disk_intermediate_threshold=cfg.disk_intermediate_threshold,
                cancel_event=cfg.cancel_event,
                pause_event=cfg.pause_event,
                target_w=run_tw,
                target_h=run_th,
            )

    return finalize_result(
        prep,
        result,
        on_progress=on_progress,
        model_label=model_name,
        out_alpha=out_alpha,
    )
