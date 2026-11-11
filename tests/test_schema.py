"""Phase 0: JobSettings and NamePattern."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from upscale.schema import (
    LANGUAGE_CATALOG,
    JobSettings,
    NamePattern,
)


def test_language_catalog_has_fifteen():
    assert len(LANGUAGE_CATALOG) == 15
    enabled = [c for c in LANGUAGE_CATALOG if c["enabled"]]
    # zh and ar are frozen layout-check locales (larger CJK text, RTL).
    assert {c["code"] for c in enabled} == {"en", "pl", "zh", "ar"}


def test_name_pattern_render():
    np = NamePattern()
    name = np.render(stem="photo", scale=4, model="RealESRGAN_x4plus", ext="png")
    assert name == "photo_4x_RealESRGAN_x4plus.png"


def test_job_settings_recipe_roundtrip(tmp_path: Path):
    js = JobSettings(scale=4, model="4x-spanx4-ch48", strength=0.8)
    path = tmp_path / "recipe.upscale.json"
    js.save_recipe(path)
    loaded = JobSettings.from_recipe_file(path)
    assert loaded.scale == 4
    assert loaded.model == "4x-spanx4-ch48"
    assert loaded.strength == 0.8
    assert json.loads(path.read_text(encoding="utf-8"))["scale"] == 4


def test_scale_and_target_mutex():
    with pytest.raises(ValueError):
        JobSettings(scale=4, target_size=2000)
