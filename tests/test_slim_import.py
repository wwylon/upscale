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
    # Snapshot and restore afterwards: later tests patch the original module objects, and a
    # leaked fresh ``upscale.pipeline`` makes ``from .pipeline import run`` bypass their patches.
    import upscale as pkg

    names = [n for n in sys.modules if n.startswith("upscale.pipeline") or n.startswith("upscale.server")]
    saved = {n: sys.modules[n] for n in names}
    saved_attrs = {a: getattr(pkg, a) for a in ("pipeline", "server") if hasattr(pkg, a)}
    for name in names:
        del sys.modules[name]
    try:
        import upscale.pipeline as pipeline
        import upscale.server as server

        assert pipeline.compute_passes(4, 4) == 1
        assert server.app.title == "Upscale"
    finally:
        for name in [n for n in sys.modules if n.startswith("upscale.pipeline") or n.startswith("upscale.server")]:
            del sys.modules[name]
        sys.modules.update(saved)
        for attr, mod in saved_attrs.items():
            setattr(pkg, attr, mod)
