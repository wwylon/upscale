"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_jobs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Per-test JOBS_ROOT + clear in-memory job maps (avoids cross-test bleed)."""
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("UPSCALE_JOBS_DIR", str(root))
    import upscale.server as server

    monkeypatch.setattr(server, "JOBS_ROOT", root)
    with server._jobs_lock:
        server._jobs.clear()
    with server._compare_lock:
        server._compare_jobs.clear()
        server._compare_runs.clear()
    yield root
