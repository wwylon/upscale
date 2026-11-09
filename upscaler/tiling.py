"""Feather-blended overlapping tiled inference with strip accumulation."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable

import numpy as np
from tqdm import tqdm

# LRU cache: (h, w, overlap, feather flags...) -> mask
_MASK_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_MASK_CACHE_MAX = 64
_MASK_CACHE_LOCK = threading.Lock()


def make_tile_weight(
    h: int,
    w: int,
    overlap: int,
    *,
    feather_top: bool = True,
    feather_bottom: bool = True,
    feather_left: bool = True,
    feather_right: bool = True,
) -> np.ndarray:
    """
    2D linear ramp weight: 0 at feathered edges, 1 at interior.
    Image-border sides should pass feather_*=False so border pixels keep weight 1.
    Returns float32 array of shape (h, w).
    """
    key = (h, w, overlap, feather_top, feather_bottom, feather_left, feather_right)
    with _MASK_CACHE_LOCK:
        cached = _MASK_CACHE.get(key)
        if cached is not None:
            _MASK_CACHE.move_to_end(key)
            return cached

    ramp_h = np.ones(h, dtype=np.float32)
    ramp_w = np.ones(w, dtype=np.float32)

    if overlap > 0:
        n_h = min(overlap, h // 2)
        n_w = min(overlap, w // 2)
        if n_h > 0:
            ramp = np.linspace(0.0, 1.0, n_h, endpoint=False, dtype=np.float32)
            if feather_top:
                ramp_h[:n_h] = ramp
            if feather_bottom:
                ramp_h[-n_h:] = ramp[::-1]
        if n_w > 0:
            ramp = np.linspace(0.0, 1.0, n_w, endpoint=False, dtype=np.float32)
            if feather_left:
                ramp_w[:n_w] = ramp
            if feather_right:
                ramp_w[-n_w:] = ramp[::-1]

    mask = np.outer(ramp_h, ramp_w)
    with _MASK_CACHE_LOCK:
        _MASK_CACHE[key] = mask
        while len(_MASK_CACHE) > _MASK_CACHE_MAX:
            _MASK_CACHE.popitem(last=False)
    return mask


def clear_mask_cache() -> None:
    with _MASK_CACHE_LOCK:
        _MASK_CACHE.clear()


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Compute tile start coordinates that cover [0, length) with given overlap."""
    step = max(1, tile - overlap)
    starts = list(range(0, length - overlap, step))
    if not starts or starts[-1] + tile < length:
        starts.append(max(0, length - tile))
    out: list[int] = []
    for s in starts:
        if not out or out[-1] != s:
            out.append(s)
    return out


class JobCancelled(Exception):
    """Raised when cancel_event is set between tiles."""


# Host RAM threshold: use memmap for the output buffer above this many bytes.
DEFAULT_MEMMAP_BYTES = 512 * 1024 * 1024  # 512 MiB


def tiled_upscale(
    img: np.ndarray,
    infer_fn: Callable[[np.ndarray], np.ndarray],
    model_scale: int,
    tile: int,
    overlap: int,
    show_progress: bool = True,
    pass_label: str = "",
    cancel_event=None,
    pause_event=None,
    on_tile: Callable[[int, int], None] | None = None,
    *,
    job_dir=None,
    pass_index: int = 0,
    resume: bool = False,
    memmap_threshold: int = DEFAULT_MEMMAP_BYTES,
) -> np.ndarray:
    """
    Upscale `img` (uint8 HxWxC) using `infer_fn` in overlapping tiles.
    Tiles are blended with an edge-aware linear feather mask.

    Uses row-strip float accumulation so peak RAM is O(width × tile × scale)
    rather than a full-frame float canvas. Unflushed overlap is carried to the
    next band.

    When ``job_dir`` is set, strip flush progress is checkpointed so a later
    call with ``resume=True`` can continue. Huge outputs allocate a memmap
    under ``job_dir`` when above ``memmap_threshold`` bytes.
    """
    H, W, C = img.shape
    out_H = H * model_scale
    out_W = W * model_scale
    out_overlap = overlap * model_scale
    out_bytes = out_H * out_W * C

    y_starts = _tile_starts(H, tile, overlap)
    x_starts = _tile_starts(W, tile, overlap)
    total_tiles = len(y_starts) * len(x_starts)

    from pathlib import Path as _Path

    jd = _Path(job_dir) if job_dir is not None else None
    if jd is not None:
        jd.mkdir(parents=True, exist_ok=True)

    # When job_dir is set, always file-back the result so strip resume can
    # reopen flushed pixels (r+). Threshold only controls end materialization.
    mmap_path = (jd / f"pass_{pass_index}_out.uint8.mmap") if jd is not None else None
    use_mmap = mmap_path is not None

    flush_row = 0
    carry_top = 0
    carry_canvas: np.ndarray | None = None
    carry_weight: np.ndarray | None = None
    start_yi = 0
    resume_ok = False

    if resume and jd is not None:
        from .checkpoint import load_strip_checkpoint

        loaded = load_strip_checkpoint(
            jd,
            expect_tile=tile,
            expect_overlap=overlap,
            expect_pass=pass_index,
            expect_out_h=out_H,
            expect_out_w=out_W,
            expect_channels=C,
            expect_model_scale=model_scale,
        )
        if loaded is not None:
            meta, extras = loaded
            flush_row = meta.flush_row
            cc = extras.get("carry_canvas")
            cw = extras.get("carry_weight")
            if cc is not None and (
                cc.ndim != 3
                or cc.shape[0] < 1
                or cc.shape[1] != out_W
                or cc.shape[2] != C
            ):
                flush_row = 0
                cc = None
                cw = None
            if flush_row > 0:
                stored = extras.get("result_mmap")
                reopen = _Path(stored) if stored else mmap_path
                if (
                    reopen is not None
                    and reopen.is_file()
                    and reopen.stat().st_size >= out_bytes
                ):
                    resume_ok = True
                    mmap_path = reopen
                    carry_canvas = cc
                    carry_weight = cw
                    carry_top = int(extras.get("carry_top") or flush_row)
                else:
                    flush_row = 0
            else:
                resume_ok = True
            if resume_ok and flush_row > 0:
                for yi, _y in enumerate(y_starts):
                    if yi + 1 < len(y_starts):
                        fe = y_starts[yi + 1] * model_scale
                    else:
                        fe = out_H
                    if fe <= flush_row:
                        start_yi = yi + 1
                    else:
                        break

    result: np.ndarray
    if use_mmap and mmap_path is not None:
        mode = "r+" if resume_ok and mmap_path.is_file() and flush_row > 0 else "w+"
        if mode == "w+" and mmap_path.is_file():
            # Truncate stale file before fresh allocate
            mmap_path.unlink(missing_ok=True)
        result = np.memmap(
            mmap_path, dtype=np.uint8, mode=mode, shape=(out_H, out_W, C)
        )
    else:
        result = np.empty((out_H, out_W, C), dtype=np.uint8)

    desc = f"Tiles{' ' + pass_label if pass_label else ''}"
    pbar = tqdm(
        total=total_tiles, desc=desc, unit="tile", disable=not show_progress, leave=False
    )
    if resume_ok and start_yi > 0:
        skipped = start_yi * len(x_starts)
        if skipped:
            pbar.update(skipped)

    for yi, y in enumerate(y_starts):
        if yi < start_yi:
            continue
        y2 = min(y + tile, H)
        oy = y * model_scale
        band_bottom = y2 * model_scale

        strip_top = min(flush_row, oy) if carry_canvas is None else carry_top
        strip_bottom = band_bottom
        strip_h = strip_bottom - strip_top

        canvas = np.zeros((strip_h, out_W, C), dtype=np.float32)
        weight_map = np.zeros((strip_h, out_W, 1), dtype=np.float32)

        if carry_canvas is not None:
            ch = carry_canvas.shape[0]
            canvas[:ch] = carry_canvas
            weight_map[:ch] = carry_weight
            carry_canvas = None
            carry_weight = None

        feather_top = y > 0
        feather_bottom = y2 < H

        for xi, x in enumerate(x_starts):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled("cancelled")
            while pause_event is not None and pause_event.is_set():
                if cancel_event is not None and cancel_event.is_set():
                    raise JobCancelled("cancelled")
                import time as _time

                _time.sleep(0.05)

            x2 = min(x + tile, W)
            patch = img[y:y2, x:x2]
            upscaled = infer_fn(patch)
            if on_tile is not None:
                tile_i = yi * len(x_starts) + xi
                on_tile(tile_i, total_tiles)

            ph = y2 - y
            pw = x2 - x
            oh = ph * model_scale
            ow = pw * model_scale
            ox = x * model_scale
            if upscaled.shape[0] != oh or upscaled.shape[1] != ow:
                raise ValueError(
                    f"infer_fn output shape {upscaled.shape[:2]} != "
                    f"expected ({oh}, {ow}) for model_scale={model_scale}"
                )

            mask = make_tile_weight(
                oh,
                ow,
                out_overlap,
                feather_top=feather_top,
                feather_bottom=feather_bottom,
                feather_left=x > 0,
                feather_right=x2 < W,
            ).reshape(oh, ow, 1)

            local_y = oy - strip_top
            canvas[local_y : local_y + oh, ox : ox + ow] += upscaled.astype(np.float32) * mask
            weight_map[local_y : local_y + oh, ox : ox + ow] += mask
            pbar.update(1)

        if yi + 1 < len(y_starts):
            finalize_end = y_starts[yi + 1] * model_scale
        else:
            finalize_end = out_H

        local_end = finalize_end - strip_top
        if local_end > 0:
            wslice = np.maximum(weight_map[:local_end], 1e-8)
            np.divide(canvas[:local_end], wslice, out=canvas[:local_end])
            result[flush_row:finalize_end] = np.clip(
                canvas[:local_end], 0, 255
            ).astype(np.uint8)
            flush_row = finalize_end
            if isinstance(result, np.memmap) or getattr(result, "filename", None):
                result.flush()

        # Carry unflushed overlap rows into the next band
        if finalize_end < strip_bottom:
            carry_top = finalize_end
            local_from = finalize_end - strip_top
            carry_canvas = canvas[local_from:].copy()
            carry_weight = weight_map[local_from:].copy()
        else:
            carry_canvas = None
            carry_weight = None

        # Checkpoint periodically (every 2 bands + first/last) to cut disk I/O.
        if job_dir is not None and (
            yi == 0
            or yi == len(y_starts) - 1
            or (yi + 1) % 2 == 0
            or (pause_event is not None and pause_event.is_set())
        ):
            from .checkpoint import StripCheckpoint, save_strip_checkpoint

            save_strip_checkpoint(
                job_dir,
                StripCheckpoint(
                    pass_index=pass_index,
                    flush_row=flush_row,
                    out_h=out_H,
                    out_w=out_W,
                    channels=C,
                    model_scale=model_scale,
                    tile=tile,
                    overlap=overlap,
                ),
                result_mmap_path=mmap_path,
                carry_canvas=carry_canvas,
                carry_weight=carry_weight,
                carry_top=carry_top if carry_canvas is not None else flush_row,
            )

        del canvas, weight_map

    pbar.close()
    if job_dir is not None:
        from .checkpoint import clear_strip_checkpoint

        clear_strip_checkpoint(job_dir)
    if isinstance(result, np.memmap) or getattr(result, "filename", None) is not None:
        result.flush()
        # Keep the memmap for mid-size results when job_dir owns it — avoids
        # doubling peak RAM. Callers that need a plain ndarray can copy.
        if out_bytes < memmap_threshold * 4 and jd is None:
            result = np.array(result)
    return result


def make_infer_fn(descriptor, device, fp16: bool):
    """
    Returns a callable (uint8 HxWxC np.ndarray) -> (uint8 HxWxC np.ndarray).
    Handles tensor conversion and dtype. OOM retry lives in pipeline._upscale_pass.
    Torch is imported lazily so the ONNX/slim desktop path can use tiled_upscale
    without shipping PyTorch.
    """
    import torch

    model = descriptor.model
    use_half = bool(fp16 and getattr(device, "type", None) == "cuda")
    dtype = torch.float16 if use_half else torch.float32

    def infer(patch: np.ndarray) -> np.ndarray:
        from .infer_util import nchw_float_to_uint8, patch_to_nchw_float

        nchw = patch_to_nchw_float(patch)
        t = torch.from_numpy(nchw).to(device=device, dtype=dtype, non_blocking=True)

        with torch.no_grad():
            out = model(t)
            out = out.float().clamp_(0, 1)
            # Quantize on device so D2H is uint8-sized after round-trip via float
            # host path kept for dtype compatibility with nchw_float_to_uint8.
            host = out.cpu().numpy()

        result = nchw_float_to_uint8(host)
        del t, out
        return result

    return infer
