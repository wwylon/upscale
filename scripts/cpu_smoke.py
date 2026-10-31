#!/usr/bin/env python3
"""Agent/CPU smoke: classical 4× + API estimate (no GPU/models required)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from upscaler.config import UpscaleConfig
    from upscaler.engine import execute
    from upscaler.estimates import estimate_time_seconds

    with tempfile.TemporaryDirectory(prefix="upscale-smoke-") as td:
        td_path = Path(td)
        src = td_path / "in.png"
        out = td_path / "out.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(src)
        actual = execute(
            UpscaleConfig(
                input=src,
                output=out,
                scale=4.0,
                algorithm="nearest",
                quiet=True,
            )
        )
        assert actual.is_file()
        arr = np.array(Image.open(actual))
        assert arr.shape[0] == 128 and arr.shape[1] == 128
        lo, hi = estimate_time_seconds(32, 32, scale=4, model="RealESRGAN_x4plus", device="cpu")
        assert hi >= lo > 0
        print(f"OK classical 4× → {actual}  ETA cpu ~{lo}-{hi}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
