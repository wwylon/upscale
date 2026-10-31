"""Model status labels are on-disk states, not GPU-loaded."""

from __future__ import annotations

from pathlib import Path

from upscaler.models import model_status


def test_model_status_unknown_is_failed():
    assert model_status("__no_such_model_xyz__") == "failed"


def test_model_status_builtin_not_downloaded(tmp_path: Path, monkeypatch):
    from upscaler import models as m

    monkeypatch.setattr(m, "DEFAULT_WEIGHTS_DIR", tmp_path)
    st = model_status("RealESRGAN_x4plus", tmp_path)
    assert st in ("not_downloaded", "installed")
    # Fresh empty dir → not on disk
    assert st == "not_downloaded"
