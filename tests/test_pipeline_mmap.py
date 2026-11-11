"""Disk intermediate spill uses memmap (Phase 6 / P1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def test_mid_pass_spill_uses_mmap(monkeypatch, tmp_path: Path):
    """Multi-pass with low spill threshold writes a uint8 memmap spill file."""
    from upscale import pipeline

    spill_writes: list[Path] = []
    real_memmap = np.memmap

    def tracking_memmap(*args, **kwargs):
        path = Path(args[0] if args else kwargs.get("filename"))
        mode = kwargs.get("mode")
        if mode is None and len(args) > 2:
            mode = args[2]
        if mode == "w+":
            spill_writes.append(path)
        return real_memmap(*args, **kwargs)

    def fake_pass(img, *args, **kwargs):
        return np.repeat(np.repeat(img, 4, axis=0), 4, axis=1)

    monkeypatch.setattr(np, "memmap", tracking_memmap)
    monkeypatch.setattr(pipeline, "_upscale_pass", fake_pass)
    monkeypatch.setattr(pipeline, "make_infer_fn", lambda *a, **k: (lambda x: x))

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    out = pipeline.run(
        img,
        scale=16.0,  # two 4× passes
        descriptor=object(),
        device=torch.device("cpu"),
        fp16=False,
        tile=32,
        overlap=0,
        show_progress=False,
        model_scale=4,
        disk_intermediate_threshold=1,  # always spill after pass 1
    )
    assert out.shape[0] == 512 and out.shape[1] == 512
    assert spill_writes, "expected at least one memmap spill write"
    assert any("pass_" in p.name and p.suffix == ".mmap" for p in spill_writes)
