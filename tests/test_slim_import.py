"""B1: server import path must not require torch when UPSCALE_BACKEND=onnx."""

from __future__ import annotations

import sys
from types import ModuleType


def test_server_imports_without_torch(monkeypatch):
    monkeypatch.setenv("UPSCALE_BACKEND", "onnx")
    # Block torch if not already imported; if already imported, still exercise onnx backend path.
    if "torch" not in sys.modules:
        blocker = ModuleType("torch")

        def _fail(*_a, **_k):
            raise ImportError("torch blocked for slim import test")

        blocker.__getattr__ = lambda _n: (_ for _ in ()).throw(ImportError("blocked"))  # type: ignore[attr-defined]
        sys.modules["torch"] = blocker

    # Fresh import of modules that used to pull torch at top level
    for name in list(sys.modules):
        if name.startswith("upscaler.pipeline") or name.startswith("upscaler.server"):
            del sys.modules[name]

    import upscaler.pipeline as pipeline
    import upscaler.server as server

    assert pipeline.compute_passes(4, 4) == 1
    assert server.app.title == "Upscale"
