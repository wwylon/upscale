"""Compare + zip API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from upscaler.server import app

from png_fixtures import TINY_PNG


@pytest.fixture
def client():
    return TestClient(app)


def _wait_compare(client: TestClient, run_id: str, timeout_s: float = 5.0) -> dict:
    import time

    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        st = client.get(f"/api/compare/{run_id}/status")
        assert st.status_code == 200, st.text
        last = st.json()
        if last.get("state") in ("done", "cancelled", "error"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"compare {run_id} did not finish: {last}")


def test_api_compare_mocked(client: TestClient, tmp_path: Path):
    def fake_compare(image, out_dir, **kwargs):
        variants = kwargs.get("variants") or []
        out = []
        for v in variants:
            vid = v["id"]
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in vid)
            dest = Path(out_dir) / f"compare_{safe}.png"
            dest.write_bytes(TINY_PNG)
            out.append(
                {
                    "id": vid,
                    "model": vid,
                    "label": v.get("label") or vid,
                    "kind": v.get("kind"),
                    "path": str(dest),
                    "seconds": 0.1,
                    "error": None,
                }
            )
        return out

    with patch("upscaler.compare.compare_variants", side_effect=fake_compare):
        r = client.post(
            "/api/compare",
            files={"file": ("t.png", TINY_PNG, "image/png")},
            data={
                "body": json.dumps(
                    {
                        "models": ["RealESRGAN_x4plus"],
                        "algorithms": ["lanczos3"],
                        "filters": ["none", "edge"],
                        "crop": {"x": 0, "y": 0, "w": 1, "h": 1},
                        "scale": 2,
                    }
                )
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"]
    assert body.get("state") == "running"
    done = _wait_compare(client, body["id"])
    assert done["state"] == "done"
    ids = {x["model"] for x in done["results"]}
    assert "RealESRGAN_x4plus" in ids
    assert "algo:lanczos3" in ids
    assert "filt:edge" in ids
    img = client.get(f"/api/compare/{body['id']}/algo:lanczos3")
    assert img.status_code == 200


def test_api_compare_cancel(client: TestClient):
    import threading
    import time

    started = threading.Event()

    def slow_compare(image, out_dir, **kwargs):
        cancel_event = kwargs.get("cancel_event")
        started.set()
        for _ in range(100):
            if cancel_event is not None and cancel_event.is_set():
                return []
            time.sleep(0.05)
        return []

    with patch("upscaler.compare.compare_variants", side_effect=slow_compare):
        r = client.post(
            "/api/compare",
            files={"file": ("t.png", TINY_PNG, "image/png")},
            data={
                "body": json.dumps(
                    {
                        "models": ["RealESRGAN_x4plus", "4x-UltraSharp"],
                        "crop": {"x": 0, "y": 0, "w": 1, "h": 1},
                        "scale": 2,
                    }
                )
            },
        )
        assert r.status_code == 200, r.text
        run_id = r.json()["id"]
        assert started.wait(2.0)
        cancel = client.post(f"/api/compare/{run_id}/cancel")
        assert cancel.status_code == 200
        done = _wait_compare(client, run_id)
        assert done["state"] == "cancelled"


def test_build_variants_mix():
    from upscaler.compare import build_variants

    vs = build_variants(
        models=["RealESRGAN_x4plus"],
        algorithms=["nearest"],
        filters=["denoise", "none"],
        filter_amounts={"denoise": 0.4},
    )
    assert len(vs) == 4
    filt = next(v for v in vs if v["id"] == "filt:denoise")
    assert filt["filters"].pre_denoise == 0.4
    assert next(v for v in vs if v["id"] == "filt:none")["filters"].pre_denoise == 0.0


def test_api_zip_empty(client: TestClient):
    r = client.post("/api/zip", json={"job_ids": ["nope"]})
    assert r.status_code == 404


def test_api_zip_from_disk(client: TestClient, tmp_path: Path, monkeypatch):
    """Zip succeeds using on-disk result.* even when job is not in memory."""
    import upscaler.server as srv

    monkeypatch.setattr(srv, "JOBS_ROOT", tmp_path)
    jid = "a" * 32
    job_dir = tmp_path / jid
    job_dir.mkdir()
    result = job_dir / "result.png"
    result.write_bytes(TINY_PNG)
    r = client.post("/api/zip", json={"job_ids": [jid]})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/zip")
    assert len(r.content) > 32
