"""Strip checkpoints for resumable long jobs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class StripCheckpoint:
    """Flush-row progress for strip-based tiled_upscale."""

    pass_index: int
    flush_row: int
    out_h: int
    out_w: int
    channels: int
    model_scale: int
    tile: int
    overlap: int


def checkpoint_dir(job_dir: Path | str) -> Path:
    d = Path(job_dir) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_checkpoints(job_dir: Path | str) -> list[dict[str, Any]]:
    root = checkpoint_dir(job_dir)
    items: list[dict[str, Any]] = []
    strip = root / "strip.json"
    if strip.is_file():
        items.append(json.loads(strip.read_text(encoding="utf-8")))
    return items


def _fsync_path(path: Path) -> None:
    try:
        with open(path, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass


def save_strip_checkpoint(
    job_dir: Path | str,
    meta: StripCheckpoint,
    *,
    result_mmap_path: Path | None = None,
    carry_canvas: np.ndarray | None = None,
    carry_weight: np.ndarray | None = None,
    carry_top: int = 0,
) -> Path:
    """Save flush_row + optional carry buffers after a strip band.

    Carry is written first (temp → replace), then strip.json is replaced
    atomically so a crash cannot leave metadata ahead of the carry file.
    """
    root = checkpoint_dir(job_dir)
    meta_path = root / "strip.json"
    carry_path = root / "strip_carry.npz"
    payload = asdict(meta)
    payload["carry_top"] = carry_top
    payload["result_mmap"] = str(result_mmap_path) if result_mmap_path else None
    payload["has_carry"] = carry_canvas is not None and carry_weight is not None

    if carry_canvas is not None and carry_weight is not None:
        tmp_carry = root / "strip_carry.npz.tmp"
        # Uncompressed .npy archive — faster than zip on the hot path.
        np.savez(
            tmp_carry,
            canvas=carry_canvas,
            weight=carry_weight,
            carry_top=np.array([carry_top], dtype=np.int64),
        )
        # np.savez may append .npz
        if not tmp_carry.is_file() and (root / "strip_carry.npz.tmp.npz").is_file():
            (root / "strip_carry.npz.tmp.npz").replace(tmp_carry)
        _fsync_path(tmp_carry)
        os.replace(tmp_carry, carry_path)
        _fsync_path(carry_path)
    elif carry_path.is_file():
        carry_path.unlink(missing_ok=True)

    tmp_meta = root / "strip.json.tmp"
    tmp_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _fsync_path(tmp_meta)
    os.replace(tmp_meta, meta_path)
    return meta_path


def load_strip_checkpoint(
    job_dir: Path | str,
    *,
    expect_tile: int | None = None,
    expect_overlap: int | None = None,
    expect_pass: int | None = None,
    expect_out_h: int | None = None,
    expect_out_w: int | None = None,
    expect_channels: int | None = None,
    expect_model_scale: int | None = None,
) -> tuple[StripCheckpoint, dict[str, Any]] | None:
    """Load strip checkpoint; return None if missing or inconsistent."""
    root = checkpoint_dir(job_dir)
    meta_path = root / "strip.json"
    if not meta_path.is_file():
        return None
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = StripCheckpoint(
            pass_index=int(raw["pass_index"]),
            flush_row=int(raw["flush_row"]),
            out_h=int(raw["out_h"]),
            out_w=int(raw["out_w"]),
            channels=int(raw["channels"]),
            model_scale=int(raw["model_scale"]),
            tile=int(raw["tile"]),
            overlap=int(raw["overlap"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if expect_tile is not None and meta.tile != expect_tile:
        return None
    if expect_overlap is not None and meta.overlap != expect_overlap:
        return None
    if expect_pass is not None and meta.pass_index != expect_pass:
        return None
    if expect_out_h is not None and meta.out_h != expect_out_h:
        return None
    if expect_out_w is not None and meta.out_w != expect_out_w:
        return None
    if expect_channels is not None and meta.channels != expect_channels:
        return None
    if expect_model_scale is not None and meta.model_scale != expect_model_scale:
        return None

    extras: dict[str, Any] = {
        "carry_top": int(raw.get("carry_top") or 0),
        "result_mmap": raw.get("result_mmap"),
        "carry_canvas": None,
        "carry_weight": None,
        "has_carry": bool(raw.get("has_carry")),
    }
    carry_path = root / "strip_carry.npz"
    needs_carry = bool(raw.get("has_carry")) or meta.flush_row > 0
    if carry_path.is_file():
        try:
            with np.load(carry_path) as z:
                extras["carry_canvas"] = np.array(z["canvas"])
                extras["carry_weight"] = np.array(z["weight"])
                if "carry_top" in z.files:
                    extras["carry_top"] = int(z["carry_top"][0])
        except (OSError, ValueError, KeyError):
            return None
    elif needs_carry and meta.flush_row > 0 and meta.flush_row < meta.out_h:
        # Metadata claims progress past row 0 but carry is missing — unsafe.
        if bool(raw.get("has_carry", True)):
            return None
    return meta, extras


def clear_strip_checkpoint(job_dir: Path | str) -> None:
    root = checkpoint_dir(job_dir)
    for name in ("strip.json", "strip_carry.npz", "strip.json.tmp", "strip_carry.npz.tmp"):
        p = root / name
        if p.is_file():
            p.unlink(missing_ok=True)


def pass_complete_meta_path(job_dir: Path | str, pass_index: int) -> Path:
    return Path(job_dir) / f"pass_{pass_index}_complete.json"


def pass_complete_mmap_path(job_dir: Path | str, pass_index: int) -> Path:
    return Path(job_dir) / f"pass_{pass_index}_out.uint8.mmap"


def save_pass_complete(
    job_dir: Path | str,
    pass_index: int,
    arr: np.ndarray,
) -> Path:
    """Persist a finished pass output under the job dir for multi-pass resume."""
    jd = Path(job_dir)
    jd.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(x) for x in arr.shape)
    mmap_path = pass_complete_mmap_path(jd, pass_index)
    meta_path = pass_complete_meta_path(jd, pass_index)
    # tiled_upscale already file-backs into this path when job_dir is set —
    # flush + metadata only. Rewriting via os.replace fails on Windows while
    # the caller's memmap still holds the destination open.
    existing = getattr(arr, "filename", None)
    already = False
    if existing is not None:
        try:
            already = Path(existing).resolve() == mmap_path.resolve()
        except OSError:
            already = False
    if already:
        try:
            arr.flush()
        except Exception:  # noqa: BLE001
            pass
    else:
        tmp_mmap = jd / f"pass_{pass_index}_out.uint8.mmap.tmp"
        if tmp_mmap.is_file():
            tmp_mmap.unlink(missing_ok=True)
        # Prefer plain file I/O so Windows can replace without an open memmap handle.
        src = np.ascontiguousarray(arr, dtype=np.uint8)
        with open(tmp_mmap, "wb") as f:
            step = max(1, 64)
            for y0 in range(0, shape[0], step):
                y1 = min(shape[0], y0 + step)
                f.write(src[y0:y1].tobytes())
        if mmap_path.is_file():
            mmap_path.unlink(missing_ok=True)
        os.replace(tmp_mmap, mmap_path)
    tmp_meta = jd / f"pass_{pass_index}_complete.json.tmp"
    tmp_meta.write_text(
        json.dumps({"pass_index": pass_index, "shape": list(shape)}, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_meta, meta_path)
    return mmap_path


def load_pass_complete(
    job_dir: Path | str, pass_index: int
) -> np.ndarray | None:
    jd = Path(job_dir)
    meta_path = pass_complete_meta_path(jd, pass_index)
    mmap_path = pass_complete_mmap_path(jd, pass_index)
    if not meta_path.is_file() or not mmap_path.is_file():
        return None
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        shape = tuple(int(x) for x in raw["shape"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    need = int(np.prod(shape))
    if mmap_path.stat().st_size < need:
        return None
    return np.memmap(mmap_path, dtype=np.uint8, mode="r", shape=shape)


def first_incomplete_pass(job_dir: Path | str, passes: int) -> int:
    """Return 0-based index of the first pass without a complete output."""
    for p in range(1, passes + 1):
        if load_pass_complete(job_dir, p) is None:
            return p - 1
    return passes
