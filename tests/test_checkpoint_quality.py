from __future__ import annotations

import numpy as np

from upscaler.checkpoint import TileCheckpoint, list_checkpoints, load_tile_checkpoint, save_tile_checkpoint
from upscaler.quality import diff_metrics, strip_seam_score


def test_checkpoint_roundtrip(tmp_path):
    meta = TileCheckpoint(job_id="j1", pass_index=0, tile_index=3, width=8, height=8)
    tile = np.ones((8, 8, 3), dtype=np.float32)
    save_tile_checkpoint(tmp_path, meta, tile)
    assert len(list_checkpoints(tmp_path)) == 1
    meta2, loaded = load_tile_checkpoint(tmp_path, 0, 3)
    assert meta2.tile_index == 3
    assert np.allclose(loaded, tile)


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
