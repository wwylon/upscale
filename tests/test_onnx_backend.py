"""ONNX backend unit tests (no real model required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from upscale.backends import resolve_backend
from upscale.backends.onnx_rt import make_onnx_infer_fn, onnx_path_for
from upscale.tiling import tiled_upscale


def test_resolve_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSCALE_BACKEND", "onnx")
    assert resolve_backend() == "onnx"
    monkeypatch.setenv("UPSCALE_BACKEND", "torch")
    assert resolve_backend() == "torch"


def test_onnx_path_for_span() -> None:
    p = onnx_path_for("4x-spanx4-ch48", Path("models"))
    assert p.name == "4x-spanx4-ch48.onnx"


def test_make_onnx_infer_fn_and_tiles() -> None:
    """Fake ORT session: 4x nearest via numpy."""

    class FakeSession:
        def run(self, _outs, feeds):
            x = feeds["input"]  # 1x3xHxW
            # crude 4x upsample
            y = np.repeat(np.repeat(x, 4, axis=2), 4, axis=3)
            return [y]

        def get_inputs(self):
            class I:
                name = "input"

            return [I()]

    infer = make_onnx_infer_fn(FakeSession(), "input")
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    out = tiled_upscale(
        img,
        infer,
        model_scale=4,
        tile=16,
        overlap=4,
        show_progress=False,
    )
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8


def test_execute_dispatches_to_onnx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UPSCALE_BACKEND", "onnx")
    called = {"ok": False}

    def fake_execute_onnx(cfg, **kwargs):
        called["ok"] = True
        out = tmp_path / "out.png"
        out.write_bytes(b"x")
        return out

    monkeypatch.setattr(
        "upscale.backends.onnx_rt.execute_onnx",
        fake_execute_onnx,
    )
    from upscale.config import UpscaleConfig
    from upscale.engine import execute

    src = tmp_path / "in.png"
    # minimal PNG via PIL
    from PIL import Image

    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(src)

    result = execute(
        UpscaleConfig(
            input=src,
            output=tmp_path / "out.png",
            scale=2,
            model="4x-spanx4-ch48",
            quiet=True,
        )
    )
    assert called["ok"]
    assert Path(result).exists()
