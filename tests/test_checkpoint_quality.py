from __future__ import annotations

import numpy as np

from upscale.checkpoint import (
    StripCheckpoint,
    list_checkpoints,
    load_strip_checkpoint,
    save_strip_checkpoint,
)
from upscale.quality import diff_metrics, strip_seam_score


def test_strip_checkpoint_listed(tmp_path):
    meta = StripCheckpoint(
        pass_index=1,
        flush_row=8,
        out_h=32,
        out_w=32,
        channels=3,
        model_scale=2,
        tile=16,
        overlap=4,
    )
    carry = np.ones((4, 32, 3), dtype=np.float32)
    weight = np.ones((4, 32, 1), dtype=np.float32)
    save_strip_checkpoint(
        tmp_path, meta, carry_canvas=carry, carry_weight=weight, carry_top=8
    )
    items = list_checkpoints(tmp_path)
    assert len(items) == 1
    assert items[0]["flush_row"] == 8
    loaded = load_strip_checkpoint(tmp_path, expect_tile=16, expect_overlap=4)
    assert loaded is not None
    m2, extras = loaded
    assert m2.pass_index == 1
    assert extras["carry_canvas"].shape == carry.shape


def test_strip_checkpoint_rejects_tile_mismatch(tmp_path):
    meta = StripCheckpoint(
        pass_index=1,
        flush_row=8,
        out_h=32,
        out_w=32,
        channels=3,
        model_scale=2,
        tile=16,
        overlap=4,
    )
    save_strip_checkpoint(tmp_path, meta, carry_top=8)
    assert load_strip_checkpoint(tmp_path, expect_tile=8) is None


def test_seam_and_diff():
    left = np.zeros((4, 10, 3), dtype=np.uint8)
    right = np.zeros((4, 10, 3), dtype=np.uint8)
    right[:, :4, :] = 255
    score = strip_seam_score(left, right, overlap=4)
    assert score > 100
    same = np.zeros((4, 4, 3), dtype=np.uint8)
    m = diff_metrics(same, same.copy())
    assert m["mae"] == 0.0
    assert m["psnr"] >= 90
