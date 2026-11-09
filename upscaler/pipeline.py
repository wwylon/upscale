"""Multi-pass upscale pipeline: pass calculation, pass loop, final resize."""

from __future__ import annotations

import gc
import importlib.util
import math
import tempfile
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from .io_utils import lanczos_resize
from .tiling import JobCancelled, make_infer_fn, tiled_upscale

DEFAULT_DISK_INTERMEDIATE_THRESHOLD = 16000


def resolve_model_scale(descriptor, fallback: int = 4) -> int:
    """Read scale from spandrel descriptor, else fallback."""
    scale = getattr(descriptor, "scale", None)
    if scale is None:
        return fallback
    try:
        return int(scale)
    except (TypeError, ValueError):
        return fallback


def compute_passes(scale: float, model_scale: int = 4) -> int:
    if scale <= 1.0:
        return 0
    ms = max(2, int(model_scale))
    floor_passes = max(1, int(math.log(scale, ms)))
    intermediate = ms**floor_passes
    remaining = scale / intermediate
    if remaining <= 1.5:
        return floor_passes
    return floor_passes + 1


def _is_memmap(x: np.ndarray) -> bool:
    # Prefer isinstance when np.memmap is still a type (tests may monkeypatch it).
    mm = getattr(np, "memmap", None)
    if isinstance(mm, type):
        try:
            if isinstance(x, mm):
                return True
        except TypeError:
            pass
    return getattr(x, "filename", None) is not None or type(x).__name__ == "memmap"


def _materialize(arr: np.ndarray) -> np.ndarray:
    if _is_memmap(arr):
        return np.array(arr, copy=True)
    return np.asarray(arr)


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    if "out of memory" in msg or "oom" in msg:
        return True
    if _torch_available():
        import torch

        if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
            return True
    return False


def run(
    img: np.ndarray,
    scale: float,
    descriptor,
    device,
    fp16: bool,
    tile: int,
    overlap: int,
    show_progress: bool = True,
    model_scale: int | None = None,
    disk_intermediate_threshold: int = DEFAULT_DISK_INTERMEDIATE_THRESHOLD,
    progress_callback: Callable[[str, float], None] | None = None,
    cancel_event=None,
    pause_event=None,
    strength: float = 1.0,
    job_dir: Path | None = None,
    resume_from_checkpoint: bool = False,
    mid_pass_previews: bool = False,
    on_progress: Callable[[dict], None] | None = None,
    target_w: int | None = None,
    target_h: int | None = None,
    infer_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    strength_source: np.ndarray | None = None,
) -> np.ndarray:
    """
    Upscale `img` by `scale` using chained SR passes at `model_scale`.
    Final Lanczos resize brings the output to the exact target dimensions.

    Strength blend (if strength < 1) uses ``strength_source`` or ``img``.
    Callers that unwrap seamless / apply post-filters themselves should pass
    ``strength=1.0`` and blend in finalize instead.
    """
    # Lazy import avoids config → pipeline → job_prep → config cycle.
    from .job_prep import apply_strength, clamp_overlap

    H, W = img.shape[:2]
    tw = target_w if target_w is not None else round(W * scale)
    th = target_h if target_h is not None else round(H * scale)
    overlap = clamp_overlap(overlap, tile)
    src_for_strength = strength_source if strength_source is not None else img

    if scale < 1.0:
        if show_progress:
            print(f"  Downscale {W}×{H} → {tw}×{th} (Lanczos)")
        out = lanczos_resize(img, tw, th)
        return apply_strength(out, src_for_strength, tw, th, strength)

    if scale == 1.0:
        out = img.copy()
        if out.shape[1] != tw or out.shape[0] != th:
            out = lanczos_resize(out, tw, th)
        return apply_strength(out, src_for_strength, tw, th, strength)

    ms = model_scale if model_scale is not None else resolve_model_scale(descriptor)
    passes = compute_passes(scale, ms)

    if passes == 0:
        out = (
            lanczos_resize(img, tw, th)
            if (img.shape[1] != tw or img.shape[0] != th)
            else img.copy()
        )
        return apply_strength(out, src_for_strength, tw, th, strength)

    if infer_fn is None:
        if descriptor is None:
            raise ValueError("descriptor or infer_fn required for SR passes")
        infer_fn = make_infer_fn(descriptor, device, fp16)

    if show_progress:
        intermediate_w = W * (ms**passes)
        intermediate_h = H * (ms**passes)
        print(
            f"  {W}×{H}  →  {tw}×{th}  "
            f"({passes} pass{'es' if passes != 1 else ''} @ {ms}x, "
            f"intermediate {intermediate_w}×{intermediate_h})"
        )

    if _torch_available() and getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)
        free_mb = (torch.cuda.mem_get_info(device)[0]) // (1024 * 1024)
        if show_progress:
            print(f"  Free VRAM before run: {free_mb} MB")

    t0 = time.perf_counter()
    current = img
    tmp_dir: tempfile.TemporaryDirectory | None = None
    if job_dir is not None:
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

    start_pass = 0
    if resume_from_checkpoint and job_dir is not None:
        from .checkpoint import first_incomplete_pass, load_pass_complete

        start_pass = first_incomplete_pass(job_dir, passes)
        if start_pass >= passes:
            # All passes already complete — load the last output and skip SR.
            loaded = load_pass_complete(job_dir, passes)
            if loaded is not None:
                current = loaded
            start_pass = passes
        elif start_pass > 0:
            loaded = load_pass_complete(job_dir, start_pass)
            if loaded is not None:
                current = loaded
            else:
                start_pass = 0
                current = img

    pass_bar = tqdm(
        range(start_pass, passes),
        desc="Passes",
        unit="pass",
        disable=not show_progress,
        position=0,
        initial=start_pass,
        total=passes,
    )

    try:
        for p in pass_bar:
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled("cancelled")
            h, w = current.shape[:2]
            pass_bar.set_description(
                f"Pass {p + 1}/{passes}  ({w}×{h} → {w * ms}×{h * ms})"
            )
            if progress_callback:
                progress_callback(f"pass {p + 1}/{passes}", p / passes)
            if on_progress:
                on_progress(
                    {
                        "type": "progress",
                        "pass": p + 1,
                        "passes": passes,
                        "pct": p / passes,
                        "message": f"pass {p + 1}/{passes}",
                    }
                )

            def _tile_cb(ti: int, tt: int, _p=p, _passes=passes):
                if on_progress:
                    base = _p / _passes
                    span = 1.0 / _passes
                    on_progress(
                        {
                            "type": "tile",
                            "pass": _p + 1,
                            "passes": _passes,
                            "tile": ti + 1,
                            "tiles": tt,
                            "pct": base + span * ((ti + 1) / max(tt, 1)),
                        }
                    )

            # Resume strip only for the first incomplete pass.
            do_resume = bool(
                resume_from_checkpoint and job_dir is not None and p == start_pass
            )
            prev = current
            current = _upscale_pass(
                prev,
                infer_fn,
                tile,
                overlap,
                show_progress,
                f"(pass {p + 1}/{passes})",
                model_scale=ms,
                cancel_event=cancel_event,
                pause_event=pause_event,
                on_tile=_tile_cb if on_progress else None,
                device=device,
                job_dir=job_dir,
                pass_index=p + 1,
                resume=do_resume,
            )
            del prev

            if job_dir is not None:
                from .checkpoint import save_pass_complete

                save_pass_complete(job_dir, p + 1, current)

            if mid_pass_previews and job_dir is not None:
                from PIL import Image

                preview = current
                ph, pw = preview.shape[:2]
                if max(ph, pw) > 1024:
                    r = 1024 / max(ph, pw)
                    # Strip-friendly: materialize a small preview only.
                    preview = lanczos_resize(
                        _materialize(preview),
                        max(1, round(pw * r)),
                        max(1, round(ph * r)),
                    )
                else:
                    preview = _materialize(preview)
                art = job_dir / f"pass_{p + 1}.png"
                Image.fromarray(preview).save(art)
                if on_progress:
                    on_progress(
                        {
                            "type": "pass_complete",
                            "pass": p + 1,
                            "passes": passes,
                            "pct": (p + 1) / passes,
                            "artifact": str(art),
                        }
                    )

            long_edge = max(current.shape[:2])
            if (
                p < passes - 1
                and disk_intermediate_threshold > 0
                and long_edge >= disk_intermediate_threshold
            ):
                if job_dir is not None and _is_memmap(current):
                    # Already file-backed under job_dir — keep it as next input.
                    pass
                else:
                    spill_root = job_dir
                    if spill_root is None:
                        if tmp_dir is None:
                            tmp_dir = tempfile.TemporaryDirectory(prefix="upscale_")
                        spill_root = Path(tmp_dir.name)
                    spill_path = Path(spill_root) / f"pass_{p + 1}_spill.uint8.mmap"
                    arr = _materialize(current)
                    shape = tuple(arr.shape)
                    mm = np.memmap(spill_path, dtype=np.uint8, mode="w+", shape=shape)
                    step = max(1, 64)
                    for y0 in range(0, shape[0], step):
                        y1 = min(shape[0], y0 + step)
                        mm[y0:y1] = arr[y0:y1]
                    mm.flush()
                    del current, arr, mm
                    current = np.memmap(
                        spill_path, dtype=np.uint8, mode="r", shape=shape
                    )

        pass_bar.set_description("Done")
        pass_bar.close()

        out_h, out_w = current.shape[:2]
        if out_w != tw or out_h != th:
            if show_progress:
                print(f"  Resize {out_w}×{out_h} → {tw}×{th} (Lanczos)")
            current = lanczos_resize(_materialize(current), tw, th)
        elif _is_memmap(current):
            current = np.array(current, copy=True)

        current = apply_strength(current, src_for_strength, tw, th, strength)

        elapsed = time.perf_counter() - t0
        vram_info = ""
        if _torch_available() and getattr(device, "type", None) == "cuda":
            import torch

            peak_mb = torch.cuda.max_memory_allocated(device) // (1024 * 1024)
            vram_info = f", peak VRAM {peak_mb} MB"
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

        if show_progress:
            print(f"  Done in {elapsed:.1f}s{vram_info}")
        if progress_callback:
            progress_callback("done", 1.0)
        if on_progress:
            on_progress({"type": "done", "pct": 1.0, "message": "done"})

        return current
    except BaseException:
        # Release memmap before tmp cleanup on cancel/error (Windows lock).
        try:
            if _is_memmap(current):
                try:
                    current._mmap.close()  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
                current = None  # type: ignore[assignment]
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        raise
    finally:
        if tmp_dir is not None:
            try:
                if current is not None and _is_memmap(current):
                    try:
                        current._mmap.close()  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass
                    current = None  # type: ignore[assignment]
            except Exception:  # noqa: BLE001
                pass
            gc.collect()
            try:
                tmp_dir.cleanup()
            except OSError:
                pass


def _upscale_pass(
    img: np.ndarray,
    infer_fn,
    tile: int,
    overlap: int,
    show_progress: bool,
    pass_label: str,
    model_scale: int = 4,
    cancel_event=None,
    pause_event=None,
    on_tile=None,
    device=None,
    job_dir: Path | None = None,
    pass_index: int = 0,
    resume: bool = False,
) -> np.ndarray:
    from .job_prep import clamp_overlap

    current_tile = tile
    current_overlap = clamp_overlap(overlap, current_tile)
    # Shrink on OOM down to tile=128. Tiles already below 128 run once.
    while True:
        try:
            return tiled_upscale(
                img,
                infer_fn,
                model_scale=model_scale,
                tile=current_tile,
                overlap=current_overlap,
                show_progress=show_progress,
                pass_label=pass_label,
                cancel_event=cancel_event,
                pause_event=pause_event,
                on_tile=on_tile,
                job_dir=job_dir,
                pass_index=pass_index,
                # Resume only when tile still matches the checkpoint's tile.
                resume=resume and current_tile == tile,
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc) or current_tile <= 128:
                raise
            new_tile = max(128, current_tile // 2)
            if new_tile >= current_tile:
                raise RuntimeError("Exhausted OOM retries") from exc
            warnings.warn(
                f"OOM at tile={current_tile}; retrying with tile={new_tile}",
                stacklevel=2,
            )
            if _torch_available() and getattr(device, "type", None) == "cuda":
                import torch

                gc.collect()
                torch.cuda.empty_cache()
            else:
                gc.collect()
            current_tile = new_tile
            current_overlap = clamp_overlap(overlap, current_tile)
            # After first OOM, do not resume a mismatched strip.
            resume = False


def upscale_alpha(
    alpha: np.ndarray,
    scale: float,
    descriptor,
    device,
    fp16: bool,
    tile: int,
    overlap: int,
    show_progress: bool = False,
    model_scale: int | None = None,
    disk_intermediate_threshold: int = DEFAULT_DISK_INTERMEDIATE_THRESHOLD,
    cancel_event=None,
    pause_event=None,
    target_w: int | None = None,
    target_h: int | None = None,
    infer_fn=None,
) -> np.ndarray:
    rgb = np.stack([alpha, alpha, alpha], axis=-1)
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
        disk_intermediate_threshold=disk_intermediate_threshold,
        cancel_event=cancel_event,
        pause_event=pause_event,
        strength=1.0,
        target_w=target_w,
        target_h=target_h,
        infer_fn=infer_fn,
    )
    return out.mean(axis=-1).round().astype(np.uint8)
