"""Crop path: output dims follow crop × scale; filters still applied."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from upscale.config import UpscaleConfig
from upscale.engine import execute
from upscale.schema import FilterSettings


def _write_png(path: Path, w: int = 100, h: int = 80) -> None:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = 40
    arr[..., 1] = 80
    arr[..., 2] = 120
    Image.fromarray(arr, mode="RGB").save(path)


def test_crop_scales_region_only(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _write_png(src, 100, 80)
    crop = (10, 10, 32, 24)  # x,y,w,h

    def fake_run(img, scale, **kwargs):
        h, w = img.shape[:2]
        assert (w, h) == (32, 24)
        return np.zeros((int(h * scale), int(w * scale), 3), dtype=np.uint8)

    cfg = UpscaleConfig(
        input=src,
        output=out,
        scale=4.0,
        model="RealESRGAN_x4plus",
        device="cpu",
        tile=64,
        quiet=True,
        crop=crop,
        mid_pass_previews=False,
    )
    with (
        patch("upscale.engine.ensure_model", return_value=tmp_path / "fake.pth"),
        patch("upscale.engine.load_model", return_value=object()),
        patch("upscale.engine.resolve_model_scale", return_value=4),
        patch("upscale.engine.run", side_effect=fake_run),
    ):
        result = execute(cfg, echo=lambda _m: None)

    assert result.is_file()
    with Image.open(result) as im:
        assert im.size == (128, 96)  # 32*4 × 24*4


def test_crop_applies_pre_filters(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _write_png(src, 64, 64)
    seen = {"pre": False}

    def fake_run(img, scale, **kwargs):
        # Pre-denoise changes pixels vs solid source
        assert not np.all(img == 40) or img.mean() != 40
        seen["pre"] = True
        h, w = img.shape[:2]
        return np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    cfg = UpscaleConfig(
        input=src,
        output=out,
        scale=2.0,
        model="4x-spanx4-ch48",
        device="cpu",
        tile=32,
        quiet=True,
        crop=(0, 0, 32, 32),
        filters=FilterSettings(pre_denoise=0.8, jpeg_clean=0.5),
        mid_pass_previews=False,
    )
    with (
        patch("upscale.engine.ensure_model", return_value=tmp_path / "fake.pth"),
        patch("upscale.engine.load_model", return_value=object()),
        patch("upscale.engine.resolve_model_scale", return_value=2),
        patch("upscale.engine.run", side_effect=fake_run),
        patch("upscale.engine.compute_passes", return_value=1),
    ):
        execute(cfg, echo=lambda _m: None)
    assert seen["pre"]
