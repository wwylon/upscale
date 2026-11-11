"""Pause event blocks between tiles then resumes."""

from __future__ import annotations

import threading
import time

import numpy as np

from upscale.tiling import tiled_upscale


def test_pause_blocks_then_resumes():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    pause = threading.Event()
    calls = {"n": 0}
    resumed = {"ok": False}

    def infer(patch):
        calls["n"] += 1
        if calls["n"] == 1:
            pause.set()

            def clear_later():
                time.sleep(0.15)
                pause.clear()
                resumed["ok"] = True

            threading.Thread(target=clear_later, daemon=True).start()
        h, w = patch.shape[:2]
        return np.zeros((h * 4, w * 4, 3), dtype=np.uint8)

    out = tiled_upscale(
        img,
        infer,
        model_scale=4,
        tile=32,
        overlap=8,
        show_progress=False,
        pause_event=pause,
    )
    assert resumed["ok"]
    assert calls["n"] >= 2
    assert out.shape[0] == 256 and out.shape[1] == 256
