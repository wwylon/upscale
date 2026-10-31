"""PyInstaller runtime hook: point Upscale at bundled data dirs."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _setup() -> None:
    if not getattr(sys, "frozen", False):
        return
    # Slim installer never ships Torch
    os.environ["UPSCALE_BACKEND"] = "onnx"
    meipass = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    exe_dir = Path(sys.executable).resolve().parent
    # Datas land in _MEIPASS for onedir COLLECT as well
    for key, rel in (
        ("UPSCALE_UI_DIR", "ui"),
        ("UPSCALE_DOCS_DIR", "docs"),
        ("UPSCALE_LOCALES_DIR", "locales"),
    ):
        candidate = meipass / rel
        if not candidate.is_dir():
            candidate = exe_dir / rel
        if candidate.is_dir():
            os.environ.setdefault(key, str(candidate))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        models = Path(local) / "Upscale" / "models"
        models.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("UPSCALE_MODELS_DIR", str(models))


_setup()
