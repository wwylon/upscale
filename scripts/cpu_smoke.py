#!/usr/bin/env python3
"""CPU smoke: classical nearest 32→64 always; weight-backed pass if models/*.pth exist."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _classical(td: Path) -> Path:
    from upscale.config import UpscaleConfig
    from upscale.engine import execute

    src = td / "in.png"
    out = td / "nearest.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(src)
    actual = execute(
        UpscaleConfig(
            input=src,
            output=out,
            scale=2.0,
            algorithm="nearest",
            quiet=True,
        )
    )
    assert actual.is_file()
    arr = np.array(Image.open(actual))
    assert arr.shape == (64, 64, 3), arr.shape
    print(f"OK classical nearest 32→64 → {actual}")
    return src


def _weight_backed(root: Path, src: Path, td: Path) -> None:
    models_dir = root / "models"
    weights: list[Path] = []
    if models_dir.is_dir():
        weights = sorted(p for p in models_dir.glob("*.pth") if p.is_file())
    if not weights:
        print("SKIP weight-backed smoke (no models/*.pth)")
        return

    from upscale.config import UpscaleConfig
    from upscale.engine import execute
    from upscale.models import FAST_MODEL, MODELS

    named = [p for p in weights if p.stem in MODELS]
    preferred = models_dir / f"{FAST_MODEL}.pth"
    if preferred.is_file() and preferred.stem in MODELS:
        weight = preferred
    elif named:
        weight = min(named, key=lambda p: p.stat().st_size)
    else:
        print(f"SKIP weight-backed smoke ({weights[0].name} is not a registry model)")
        return

    name = weight.stem
    out = td / "weight.png"
    # Tile must be >= 128; smaller values never enter the tiled pass.
    actual = execute(
        UpscaleConfig(
            input=src,
            output=out,
            scale=4.0,
            model=name,
            cpu=True,
            tile=128,
            quiet=True,
        )
    )
    assert actual.is_file(), name
    arr = np.array(Image.open(actual))
    assert arr.shape[0] >= 64 and arr.shape[1] >= 64, arr.shape
    print(f"OK weight-backed {name} → {actual} {arr.shape[1]}×{arr.shape[0]}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    with tempfile.TemporaryDirectory(prefix="upscale-smoke-") as td:
        td_path = Path(td)
        src = _classical(td_path)
        _weight_backed(root, src, td_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
