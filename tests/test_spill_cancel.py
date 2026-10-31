"""Spill memmap cleanup on cancel; alpha cancel plumbing."""

from __future__ import annotations

import threading
from unittest.mock import patch

import numpy as np
import pytest
import torch

from upscaler import pipeline
from upscaler.tiling import JobCancelled


def _x4(patch: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(patch, 4, axis=0), 4, axis=1)


def test_cancel_after_spill_does_not_raise_oserror(tmp_path):
    """After pass 1 spills to mmap, cancel; finally must not raise OSError."""
    cancel = threading.Event()
    n = {"passes": 0}
    real = pipeline._upscale_pass

    def counting_pass(img, *args, **kwargs):
        n["passes"] += 1
        out = real(img, *args, **kwargs)
        if n["passes"] >= 1:
            cancel.set()
        return out

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    with patch.object(pipeline, "_upscale_pass", side_effect=counting_pass):
        with pytest.raises(JobCancelled):
            pipeline.run(
                img,
                scale=16.0,
                descriptor=None,
                device=torch.device("cpu"),
                fp16=False,
                tile=16,
                overlap=0,
                show_progress=False,
                model_scale=4,
                disk_intermediate_threshold=1,
                job_dir=tmp_path,
                infer_fn=_x4,
                cancel_event=cancel,
            )
    assert n["passes"] >= 1


def test_alpha_forwards_cancel_and_pause():
    """upscale_alpha must thread cancel/pause into run (no second stuck job)."""
    ev_c = threading.Event()
    ev_p = threading.Event()
    seen: dict = {}

    def fake_run(*_a, **kwargs):
        seen.update(kwargs)
        return np.zeros((16, 16, 3), dtype=np.uint8)

    with patch.object(pipeline, "run", side_effect=fake_run):
        out = pipeline.upscale_alpha(
            np.zeros((8, 8), dtype=np.uint8),
            scale=2.0,
            descriptor=None,
            device=torch.device("cpu"),
            fp16=False,
            tile=8,
            overlap=0,
            cancel_event=ev_c,
            pause_event=ev_p,
            infer_fn=lambda x: x,
        )
    assert out.shape == (16, 16)
    assert seen.get("cancel_event") is ev_c
    assert seen.get("pause_event") is ev_p
