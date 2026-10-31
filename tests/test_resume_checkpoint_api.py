"""Interrupted-job list + resume-from-strip-checkpoint API."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import upscaler.server as server
from png_fixtures import TINY_PNG
from upscaler.checkpoint import StripCheckpoint, save_strip_checkpoint
from upscaler.schema import JobSettings
from upscaler.server import JobRecord, JobState, app


@pytest.fixture
def client():
    return TestClient(app)


def _plant_strip(job_dir: Path) -> None:
    meta = StripCheckpoint(
        pass_index=0,
        flush_row=32,
        out_h=128,
        out_w=128,
        channels=3,
        model_scale=4,
        tile=64,
        overlap=8,
    )
    save_strip_checkpoint(job_dir, meta)


def test_interrupted_route_not_captured_as_job_id(client: TestClient):
    r = client.get("/api/jobs/interrupted")
    assert r.status_code == 200
    assert "jobs" in r.json()


def test_interrupted_skips_done_and_clears_stale_checkpoint(client: TestClient):
    """Finished jobs with leftover strip.json must not offer resume."""
    job_id = "donestaleckpt01"
    job_dir = server.JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(TINY_PNG)
    (job_dir / "result.png").write_bytes(TINY_PNG)
    (job_dir / "settings.json").write_text(
        '{"scale":2,"algorithm":"nearest"}', encoding="utf-8"
    )
    _plant_strip(job_dir)
    with server._jobs_lock:
        server._jobs[job_id] = JobRecord(
            id=job_id,
            job_dir=job_dir,
            input_path=job_dir / "input.png",
            settings=JobSettings(scale=2, algorithm="nearest"),
            state=JobState.done,
        )

    listed = client.get("/api/jobs/interrupted").json()["jobs"]
    assert all(j["id"] != job_id for j in listed)
    assert not (job_dir / "checkpoints" / "strip.json").is_file()
    with server._jobs_lock:
        server._jobs.pop(job_id, None)


def test_interrupted_skips_unreadable_input(client: TestClient):
    job_id = "badpngresume01"
    job_dir = server.JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    # Intentionally invalid PNG bytes (legacy fixture Pillow 12+ rejects)
    bad = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    (job_dir / "input.png").write_bytes(bad)
    (job_dir / "settings.json").write_text(
        '{"scale":2,"algorithm":"nearest"}', encoding="utf-8"
    )
    save_strip_checkpoint(
        job_dir,
        StripCheckpoint(
            pass_index=0,
            flush_row=32,
            out_h=128,
            out_w=128,
            channels=3,
            model_scale=4,
            tile=64,
            overlap=8,
        ),
    )
    with server._jobs_lock:
        server._jobs.pop(job_id, None)

    listed = client.get("/api/jobs/interrupted").json()["jobs"]
    assert all(j["id"] != job_id for j in listed)
    assert not (job_dir / "checkpoints" / "strip.json").is_file()


def test_list_and_resume_checkpoint(client: TestClient, tmp_path: Path):
    out = tmp_path / "out.png"
    out.write_bytes(TINY_PNG)
    started = threading.Event()
    release = threading.Event()
    runs: list[int] = []

    def fake_execute(cfg, **kwargs):
        runs.append(1)
        started.set()
        release.wait(timeout=5)
        if cfg.cancel_event is not None and cfg.cancel_event.is_set():
            from upscaler.tiling import JobCancelled

            raise JobCancelled()
        if kwargs.get("on_progress"):
            kwargs["on_progress"]({"type": "done", "pct": 1.0, "message": "done"})
        return out

    with patch("upscaler.server.execute", side_effect=fake_execute):
        r = client.post(
            "/api/jobs",
            files={"file": ("test.png", TINY_PNG, "image/png")},
            data={"settings": '{"scale": 2, "algorithm": "nearest"}'},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        assert started.wait(timeout=10)
        job_dir = server.JOBS_ROOT / job_id
        _plant_strip(job_dir)
        assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
        release.set()

        for _ in range(100):
            with server._jobs_lock:
                st = server._jobs[job_id].state
            if st == JobState.cancelled:
                break
            time.sleep(0.05)
        else:
            pytest.fail("job never cancelled")

        listed = client.get("/api/jobs/interrupted")
        assert listed.status_code == 200
        ids = {j["id"] for j in listed.json()["jobs"]}
        assert job_id in ids

        started2 = threading.Event()
        release2 = threading.Event()

        def fake_resume(cfg, **kwargs):
            started2.set()
            release2.wait(timeout=5)
            if kwargs.get("on_progress"):
                kwargs["on_progress"]({"type": "done", "pct": 1.0, "message": "done"})
            return out

        with patch("upscaler.server.execute", side_effect=fake_resume):
            res = client.post(f"/api/jobs/{job_id}/resume-checkpoint")
            assert res.status_code == 200, res.text
            assert res.json()["status"] == "queued"
            assert started2.wait(timeout=10)
            release2.set()
            for _ in range(100):
                st = client.get(f"/api/jobs/{job_id}").json()["state"]
                if st == "done":
                    break
                time.sleep(0.05)
            else:
                pytest.fail("resumed job never done")


def test_resume_checkpoint_orphaned_from_disk(client: TestClient, tmp_path: Path):
    job_id = "orphanresume01"
    job_dir = server.JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(TINY_PNG)
    (job_dir / "settings.json").write_text(
        json.dumps({"scale": 2, "algorithm": "nearest", "device": "cpu"}),
        encoding="utf-8",
    )
    _plant_strip(job_dir)
    with server._jobs_lock:
        server._jobs.pop(job_id, None)

    listed = client.get("/api/jobs/interrupted").json()["jobs"]
    assert any(j["id"] == job_id and j["state"] == "orphaned" for j in listed)

    out = tmp_path / "out.png"
    out.write_bytes(TINY_PNG)

    def fake_execute(cfg, **kwargs):
        if kwargs.get("on_progress"):
            kwargs["on_progress"]({"type": "done", "pct": 1.0, "message": "done"})
        return out

    with patch("upscaler.server.execute", side_effect=fake_execute):
        res = client.post(f"/api/jobs/{job_id}/resume-checkpoint")
        assert res.status_code == 200, res.text
        for _ in range(100):
            st = client.get(f"/api/jobs/{job_id}")
            if st.status_code == 200 and st.json().get("state") == "done":
                break
            time.sleep(0.05)
        else:
            pytest.fail("orphaned resume never done")
