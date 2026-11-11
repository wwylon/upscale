"""Seamless multi-pass unwrap + exact width×height."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from upscale.config import UpscaleConfig
from upscale.engine import execute
from upscale.filters import unwrap_seamless
from upscale.job_prep import seamless_unwrap_scale
from upscale.schema import FilterSettings


def test_seamless_unwrap_scale_uses_padded_geometry():
    content_w, content_h, pad, scale = 64, 64, 16, 16
    padded = content_w + 2 * pad
    result = np.zeros((padded * scale, padded * scale, 3), dtype=np.uint8)
    factor = seamless_unwrap_scale(result, content_w, content_h, pad)
    assert factor == scale
    unwrapped = unwrap_seamless(result, pad, factor)
    assert unwrapped.shape[0] == content_h * scale
    assert unwrapped.shape[1] == content_w * scale


def test_seamless_unwrap_non_integer_scale_margins():
    from upscale.job_prep import seamless_unwrap_margins

    content_w, content_h, pad = 100, 100, 16
    scale = 2.5
    padded = content_w + 2 * pad
    out = int(round(padded * scale))
    result = np.zeros((out, out, 3), dtype=np.uint8)
    # Mark border pad region vs interior
    result[:] = 1
    py, px = seamless_unwrap_margins(result, content_w, content_h, pad)
    assert py == int(round(pad * out / padded))
    assert px == py
    result[py:-py, px:-px] = 200
    unwrapped = unwrap_seamless(result, pad, margins=(py, px))
    assert unwrapped.shape[0] == out - 2 * py
    assert unwrapped.shape[1] == out - 2 * px
    assert int(unwrapped.min()) == 200
    # Integer factor would have cropped 32 px; real pad is 40 — verify we
    # did not leave the padded border (value 1) inside the crop.
    assert py == 40


def test_seamless_alpha_aligns_with_rgb(tmp_path: Path):
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:] = 40
    alpha = np.full((32, 32), 255, dtype=np.uint8)
    alpha[:4, :] = 0
    src = tmp_path / "rgba.png"
    Image.fromarray(np.dstack([arr, alpha])).save(src)
    out = tmp_path / "out.png"
    cfg = UpscaleConfig(
        input=src,
        output=out,
        scale=2.5,
        algorithm="nearest",
        quiet=True,
        filters=FilterSettings(seamless=True),
    )
    actual = execute(cfg)
    result = np.array(Image.open(actual).convert("RGBA"))
    assert result.shape[0] == 80 and result.shape[1] == 80
    # Top rows should stay fully transparent after unwrap + resize.
    assert int(result[:2, :, 3].max()) == 0
    assert int(result[20:60, 20:60, 3].min()) == 255


def test_seamless_wrap_roundtrip_classical(tmp_path: Path):
    # Checker that wraps: left/right edges match after seamless pad math
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :8] = 200
    arr[:, -8:] = 200
    src = tmp_path / "tex.png"
    Image.fromarray(arr).save(src)
    out = tmp_path / "out.png"
    cfg = UpscaleConfig(
        input=src,
        output=out,
        scale=2.0,
        algorithm="nearest",
        quiet=True,
        filters=FilterSettings(seamless=True),
    )
    actual = execute(cfg)
    result = np.array(Image.open(actual))
    assert result.shape[0] == 64 and result.shape[1] == 64


def test_width_height_exact_output(tmp_path: Path):
    src = tmp_path / "in.png"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(src)
    out = tmp_path / "out.png"
    cfg = UpscaleConfig(
        input=src,
        output=out,
        width=150,
        height=200,
        algorithm="nearest",
        quiet=True,
    )
    actual = execute(cfg)
    result = np.array(Image.open(actual))
    assert result.shape[1] == 150
    assert result.shape[0] == 200


def test_estimates_match_engine_wh():
    from upscale.estimates import output_dimensions

    w, h, eff = output_dimensions(100, 100, width=150, height=200)
    assert (w, h) == (150, 200)
    assert abs(eff - 2.0) < 1e-9
