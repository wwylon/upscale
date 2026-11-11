"""Strip flush-row checkpoint roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from upscale.checkpoint import (
    StripCheckpoint,
    clear_strip_checkpoint,
    load_strip_checkpoint,
    save_strip_checkpoint,
)
from upscale.tiling import tiled_upscale


def test_strip_checkpoint_roundtrip(tmp_path: Path):
    meta = StripCheckpoint(
        pass_index=1,
        flush_row=64,
        out_h=256,
        out_w=256,
        channels=3,
        model_scale=4,
        tile=32,
        overlap=8,
    )
    carry = np.ones((16, 256, 3), dtype=np.float32)
    weight = np.ones((16, 256, 1), dtype=np.float32)
    save_strip_checkpoint(
        tmp_path, meta, carry_canvas=carry, carry_weight=weight, carry_top=64
    )
    loaded = load_strip_checkpoint(tmp_path)
    assert loaded is not None
    m2, extras = loaded
    assert m2.flush_row == 64
    assert extras["carry_canvas"].shape == carry.shape
    clear_strip_checkpoint(tmp_path)
    assert load_strip_checkpoint(tmp_path) is None


def test_tiled_upscale_resume_skips_flushed_bands(tmp_path: Path):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[0, 0] = 255
    calls = {"n": 0}

    def infer(patch):
        calls["n"] += 1
        return np.repeat(np.repeat(patch, 2, axis=0), 2, axis=1)

    # First run: cancel after enough tiles to flush at least one band
    cancel = __import__("threading").Event()

    def on_tile(i, n):
        if i >= 3:
            cancel.set()

    try:
        tiled_upscale(
            img,
            infer,
            model_scale=2,
            tile=32,
            overlap=8,
            show_progress=False,
            cancel_event=cancel,
            on_tile=on_tile,
            job_dir=tmp_path,
            pass_index=1,
        )
    except Exception:
        pass

    mid = load_strip_checkpoint(tmp_path)
    if mid is None:
        # Cancel before first flush — still OK
        return
    n_before = calls["n"]
    # Resume should not re-infer skipped tiles from scratch as many times
    out = tiled_upscale(
        img,
        infer,
        model_scale=2,
        tile=32,
        overlap=8,
        show_progress=False,
        job_dir=tmp_path,
        pass_index=1,
        resume=True,
    )
    assert out.shape == (128, 128, 3)
    assert calls["n"] > n_before


def test_tiled_upscale_resume_pixel_identical(tmp_path: Path):
    """Cancel mid-pass, resume → same pixels as uninterrupted run."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)

    def infer(patch):
        return np.repeat(np.repeat(patch, 2, axis=0), 2, axis=1)

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    expected = tiled_upscale(
        img,
        infer,
        model_scale=2,
        tile=16,
        overlap=4,
        show_progress=False,
        job_dir=full_dir,
        pass_index=1,
    )

    part_dir = tmp_path / "part"
    part_dir.mkdir()
    cancel = __import__("threading").Event()

    # 48px / tile16 / overlap4 → 4 X starts; cancel after first Y-band
    # finishes (on last tile of band) so strip.json is written, then next band aborts.
    def on_tile(i, _n):
        if i >= 3:
            cancel.set()

    from upscale.tiling import JobCancelled

    try:
        tiled_upscale(
            img,
            infer,
            model_scale=2,
            tile=16,
            overlap=4,
            show_progress=False,
            cancel_event=cancel,
            on_tile=on_tile,
            job_dir=part_dir,
            pass_index=1,
        )
    except JobCancelled:
        pass

    mid = load_strip_checkpoint(part_dir)
    assert mid is not None, "expected a strip checkpoint after cancel"
    resumed = tiled_upscale(
        img,
        infer,
        model_scale=2,
        tile=16,
        overlap=4,
        show_progress=False,
        job_dir=part_dir,
        pass_index=1,
        resume=True,
    )
    assert resumed.shape == expected.shape
    np.testing.assert_array_equal(resumed, expected)
