"""Queue limit, GPU lock vs compare, benchmark sandbox."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from png_fixtures import TINY_PNG
from upscaler.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_queue_full_returns_429(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    import upscaler.server as server

    monkeypatch.setattr(server, "MAX_QUEUE", 1)
    started = threading.Event()
    release = threading.Event()

    def fake_execute(cfg, **kwargs):
        started.set()
        deadline = time.time() + 8
        while time.time() < deadline:
            if cfg.cancel_event is not None and cfg.cancel_event.is_set():
                from upscaler.tiling import JobCancelled

                raise JobCancelled()
            if release.is_set():
                break
            time.sleep(0.05)
        out = Path(cfg.output) if cfg.output else Path("out.png")
        out.write_bytes(TINY_PNG)
        return out

    with patch("upscaler.server.execute", side_effect=fake_execute):
        r1 = client.post(
            "/api/jobs",
            files={"file": ("a.png", TINY_PNG, "image/png")},
            data={"settings": '{"scale": 2, "algorithm": "nearest"}'},
        )
        assert r1.status_code == 200, r1.text
        assert started.wait(timeout=5)
        r2 = client.post(
            "/api/jobs",
            files={"file": ("b.png", TINY_PNG, "image/png")},
            data={"settings": '{"scale": 2, "algorithm": "nearest"}'},
        )
        assert r2.status_code == 429, r2.text
        release.set()
        client.post(f"/api/jobs/{r1.json()['id']}/cancel")


def test_compare_waits_on_gpu_lock(client: TestClient):
    import upscaler.server as server

    held = threading.Event()
    release = threading.Event()

    def holder():
        with server._execute_lock:
            held.set()
            release.wait(timeout=10)

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    assert held.wait(timeout=2)

    def fake_compare(*_a, **_k):
        return [
            {
                "id": "lanczos3",
                "model": "lanczos3",
                "label": "lanczos3",
                "kind": "algorithm",
                "path": "x.png",
                "seconds": 0.01,
                "error": None,
            }
        ]

    with patch("upscaler.compare.compare_variants", side_effect=fake_compare):
        r = client.post(
            "/api/compare",
            files={"file": ("t.png", TINY_PNG, "image/png")},
            data={
                "body": (
                    '{"models":[],"algorithms":["lanczos3"],'
                    '"crop":{"x":0,"y":0,"w":1,"h":1},"scale":2}'
                )
            },
        )
        assert r.status_code == 200, r.text
        run_id = r.json()["id"]
        time.sleep(0.3)
        st = client.get(f"/api/compare/{run_id}/status").json()
        assert st["state"] in ("running", "starting"), st
        release.set()
        deadline = time.time() + 5
        while time.time() < deadline:
            st = client.get(f"/api/compare/{run_id}/status").json()
            if st["state"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.05)
        assert st["state"] == "done", st
    t.join(timeout=2)


def test_benchmark_outside_jobs_root_400(client: TestClient, tmp_path: Path):
    outside = tmp_path.parent / "outside-bench.png"
    # Ensure path is outside the patched JOBS_ROOT (tmp_path/jobs).
    outside.write_bytes(TINY_PNG)
    r = client.post(
        "/api/benchmark",
        json={
            "path": str(outside.resolve()),
            "model": "RealESRGAN_x4plus",
            "device": "cpu",
            "crop_size": 8,
            "scale": 2,
        },
    )
    assert r.status_code == 400, r.text
