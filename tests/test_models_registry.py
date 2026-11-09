"""Model status labels are on-disk states, not GPU-loaded."""

from __future__ import annotations

from pathlib import Path

from upscaler.models import model_status


def test_builtin_license_fields():
    from upscaler.models import MODELS

    assert MODELS["RealESRGAN_x4plus"]["license"] == "BSD-3-Clause-like"
    assert "non-commercial" in MODELS["4x-UltraSharp"]["license"]


def test_model_status_unknown_is_failed():
    assert model_status("__no_such_model_xyz__") == "failed"


def test_model_status_builtin_not_downloaded(tmp_path: Path, monkeypatch):
    from upscaler import models as m

    monkeypatch.setattr(m, "DEFAULT_WEIGHTS_DIR", tmp_path)
    st = model_status("RealESRGAN_x4plus", tmp_path)
    assert st in ("not_downloaded", "installed")
    # Fresh empty dir → not on disk
    assert st == "not_downloaded"


def test_prune_stale_path_only_external(tmp_path: Path):
    from upscaler.models import (
        load_user_registry,
        prune_stale_external_models,
        save_user_registry,
    )

    save_user_registry(
        {
            "gone-temp": {
                "url": "",
                "path": str(tmp_path / "missing.safetensors"),
                "scale": 4,
                "sha256": None,
                "description": "stale",
            },
            "keep-url": {
                "url": "https://huggingface.co/example/resolve/main/x.pth",
                "path": str(tmp_path / "also-missing.pth"),
                "scale": 4,
                "sha256": None,
                "description": "has url",
            },
        },
        tmp_path,
    )
    removed = prune_stale_external_models(tmp_path)
    assert removed == ["gone-temp"]
    reg = load_user_registry(tmp_path)
    assert "gone-temp" not in reg
    assert "keep-url" in reg
