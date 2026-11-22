"""Alpha-channel upsampling policy shared by Torch and ONNX backends.

Default is Lanczos (classical). Neural SR of alpha is optional, Torch-only,
and expensive: the alpha plane is expanded to 3ch, run through the same
tiled SR path as RGB, then averaged back to a single channel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .io_utils import lanczos_resize


def upsample_alpha_lanczos(
    alpha: np.ndarray,
    *,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Classical Lanczos resize of a single-channel alpha plane."""
    a = np.asarray(alpha)
    if a.ndim == 3:
        a = a[:, :, 0]
    a = np.ascontiguousarray(a, dtype=np.uint8)
    if a.shape[1] == target_w and a.shape[0] == target_h:
        return a
    return lanczos_resize(a, target_w, target_h)


def upsample_alpha_sr(
    alpha: np.ndarray,
    *,
    scale: float,
    descriptor: Any,
    device: Any,
    fp16: bool,
    tile: int,
    overlap: int,
    show_progress: bool = False,
    model_scale: int | None = None,
    disk_intermediate_threshold: int | None = None,
    cancel_event: Any = None,
    pause_event: Any = None,
    target_w: int | None = None,
    target_h: int | None = None,
    infer_fn: Any = None,
) -> np.ndarray:
    """Neural SR of alpha via grayscale→RGB stack then channel mean.

    Prefer :func:`upsample_alpha_lanczos` for soft/semi-transparent edges;
    this path is for hard mattes where SR edge detail matters and the user
    opted in with ``--sr-alpha``.
    """
    from .pipeline import DEFAULT_DISK_INTERMEDIATE_THRESHOLD, run

    a = np.asarray(alpha)
    if a.ndim == 3:
        a = a[:, :, 0]
    a = np.ascontiguousarray(a, dtype=np.uint8)
    rgb = np.stack([a, a, a], axis=-1)
    thr = (
        DEFAULT_DISK_INTERMEDIATE_THRESHOLD
        if disk_intermediate_threshold is None
        else disk_intermediate_threshold
    )
    out = run(
        rgb,
        scale=scale,
        descriptor=descriptor,
        device=device,
        fp16=fp16,
        tile=tile,
        overlap=overlap,
        show_progress=show_progress,
        model_scale=model_scale,
        disk_intermediate_threshold=thr,
        cancel_event=cancel_event,
        pause_event=pause_event,
        strength=1.0,
        target_w=target_w,
        target_h=target_h,
        infer_fn=infer_fn,
    )
    return out.mean(axis=-1).round().astype(np.uint8)


def upsample_alpha(
    alpha: np.ndarray,
    *,
    scale: float,
    target_w: int,
    target_h: int,
    sr_alpha: bool = False,
    descriptor: Any = None,
    device: Any = None,
    fp16: bool = False,
    tile: int = 256,
    overlap: int = 32,
    show_progress: bool = False,
    model_scale: int | None = None,
    disk_intermediate_threshold: int | None = None,
    cancel_event: Any = None,
    pause_event: Any = None,
    infer_fn: Any = None,
    allow_sr: bool = True,
) -> np.ndarray:
    """Resolve alpha to ``target_w×target_h``.

    When ``sr_alpha`` is False, or SR is disallowed / unavailable, uses Lanczos
    to the target size. When ``sr_alpha`` and ``allow_sr`` and a descriptor or
    ``infer_fn`` is present, runs neural SR (may produce a padded canvas that
    finalize unwraps).
    """
    if (
        sr_alpha
        and allow_sr
        and scale > 1.0
        and (descriptor is not None or infer_fn is not None)
    ):
        return upsample_alpha_sr(
            alpha,
            scale=scale,
            descriptor=descriptor,
            device=device,
            fp16=fp16,
            tile=tile,
            overlap=overlap,
            show_progress=show_progress,
            model_scale=model_scale,
            disk_intermediate_threshold=disk_intermediate_threshold,
            cancel_event=cancel_event,
            pause_event=pause_event,
            target_w=target_w,
            target_h=target_h,
            infer_fn=infer_fn,
        )
    return upsample_alpha_lanczos(alpha, target_w=target_w, target_h=target_h)
