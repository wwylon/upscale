"""Strip and tile checkpoints for resumable long jobs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TileCheckpoint:
    """Deprecated: prefer strip checkpoints (`StripCheckpoint`) for resume.

    Kept for unit-test compatibility with older tile-index metadata files.
    """

    job_id: str
    pass_index: int
    tile_index: int
    width: int
    height: int
    dtype: str = "float32"

    def filename(self) -> str:
        return f"ckpt_p{self.pass_index}_t{self.tile_index}.npz"


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


def save_tile_checkpoint(
    job_dir: Path | str,
    meta: TileCheckpoint,
    tile_data: np.ndarray,
) -> Path:
    """Persist one tile buffer plus sidecar metadata (legacy/experimental)."""
    root = checkpoint_dir(job_dir)
    meta_path = root / f"{meta.filename()}.json"
    data_path = root / meta.filename()
    meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    np.savez_compressed(data_path, tile=tile_data)
    return data_path


def load_tile_checkpoint(
    job_dir: Path | str, pass_index: int, tile_index: int
) -> tuple[TileCheckpoint, np.ndarray]:
    root = checkpoint_dir(job_dir)
    pattern = f"ckpt_p{pass_index}_t{tile_index}.npz"
    data_path = root / pattern
    meta_path = root / f"{pattern}.json"
    if not data_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"No checkpoint for pass {pass_index} tile {tile_index}")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = TileCheckpoint(**raw)
    with np.load(data_path) as z:
        tile = z["tile"]
    return meta, np.array(tile)


def list_checkpoints(job_dir: Path | str) -> list[dict[str, Any]]:
    root = checkpoint_dir(job_dir)
    items: list[dict[str, Any]] = []
    for meta_path in sorted(root.glob("*.npz.json")):
        items.append(json.loads(meta_path.read_text(encoding="utf-8")))
    strip = root / "strip.json"
    if strip.is_file():
        items.append(json.loads(strip.read_text(encoding="utf-8")))
    return items


def save_strip_checkpoint(
    job_dir: Path | str,
    meta: StripCheckpoint,
    *,
    result_mmap_path: Path | None = None,
    carry_canvas: np.ndarray | None = None,
    carry_weight: np.ndarray | None = None,
    carry_top: int = 0,
) -> Path:
    """Save flush_row + optional carry buffers after a strip band."""
    root = checkpoint_dir(job_dir)
    meta_path = root / "strip.json"
    payload = asdict(meta)
    payload["carry_top"] = carry_top
    payload["result_mmap"] = str(result_mmap_path) if result_mmap_path else None
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if carry_canvas is not None and carry_weight is not None:
        np.savez_compressed(
            root / "strip_carry.npz",
            canvas=carry_canvas,
            weight=carry_weight,
            carry_top=np.array([carry_top], dtype=np.int64),
        )
    elif (root / "strip_carry.npz").is_file():
        (root / "strip_carry.npz").unlink(missing_ok=True)
    return meta_path


def load_strip_checkpoint(
    job_dir: Path | str,
) -> tuple[StripCheckpoint, dict[str, Any]] | None:
    root = checkpoint_dir(job_dir)
    meta_path = root / "strip.json"
    if not meta_path.is_file():
        return None
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
    extras: dict[str, Any] = {
        "carry_top": int(raw.get("carry_top") or 0),
        "result_mmap": raw.get("result_mmap"),
        "carry_canvas": None,
        "carry_weight": None,
    }
    carry_path = root / "strip_carry.npz"
    if carry_path.is_file():
        with np.load(carry_path) as z:
            extras["carry_canvas"] = np.array(z["canvas"])
            extras["carry_weight"] = np.array(z["weight"])
            if "carry_top" in z.files:
                extras["carry_top"] = int(z["carry_top"][0])
    return meta, extras


def clear_strip_checkpoint(job_dir: Path | str) -> None:
    root = checkpoint_dir(job_dir)
    for name in ("strip.json", "strip_carry.npz"):
        p = root / name
        if p.is_file():
            p.unlink(missing_ok=True)
