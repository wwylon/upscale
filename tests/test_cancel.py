from __future__ import annotations

import threading

import numpy as np
import pytest

from upscaler.tiling import JobCancelled, tiled_upscale


def test_cancel_between_tiles():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cancel = threading.Event()
    calls = {"n": 0}

    def infer(patch):
        calls["n"] += 1
        if calls["n"] >= 2:
            cancel.set()
        h, w = patch.shape[:2]
        return np.zeros((h * 4, w * 4, 3), dtype=np.uint8)

    with pytest.raises(JobCancelled):
        tiled_upscale(
            img,
            infer,
            model_scale=4,
            tile=32,
            overlap=8,
            show_progress=False,
            cancel_event=cancel,
        )
