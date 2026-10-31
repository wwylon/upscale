from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from upscaler.presets import list_presets, resolve_option_preset, resolve_target_preset
from upscaler.suggest import analyze_image, classify_kind, suggest_model


@pytest.fixture
def gray_png(tmp_path: Path) -> Path:
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    p = tmp_path / "gray.png"
    Image.fromarray(arr).save(p)
    return p


@pytest.fixture
def noisy_png(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    arr = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
    p = tmp_path / "noise.png"
    Image.fromarray(arr).save(p)
    return p


def test_target_presets():
    p = resolve_target_preset("4k")
    assert p.target_size == 3840
    data = list_presets()
    assert any(x["id"] == "photo_4x" for x in data["options"])


def test_option_preset_photo():
    opt = resolve_option_preset("photo_4x")
    assert opt.settings["scale"] == 4.0


def test_suggest_flat(gray_png: Path):
    out = suggest_model(gray_png)
    assert out["model"]
    assert out["kind"] in ("flat", "photo", "art", "noisy")


def test_classify_noisy(noisy_png: Path):
    stats = analyze_image(noisy_png)
    assert classify_kind(stats) == "noisy"
