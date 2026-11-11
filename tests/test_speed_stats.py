"""Learned MP/s from past runs feed time estimates."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from upscale import speed_stats as ss
from upscale.estimates import estimate_time_seconds
from upscale.server import app


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def stats_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ss, "user_config_dir", lambda: tmp_path)
    monkeypatch.setattr(ss, "speed_stats_path", lambda: tmp_path / "speed_stats.json")
    return tmp_path


def test_work_megapixels_neural_vs_classical():
    neural = ss.work_megapixels(256, 256, 4.0, model_scale=4)
    classical = ss.work_megapixels(256, 256, 4.0, algorithm="lanczos3")
    assert neural > 0
    assert classical == pytest.approx((1024 * 1024) / 1e6)


def test_record_and_lookup(stats_dir: Path):
    assert (
        ss.record_from_job(
            model="RealESRGAN_x4plus",
            device="cuda",
            seconds=10.0,
            src_w=512,
            src_h=512,
            scale=4.0,
            half=False,
        )
        is not None
    )
    hit = ss.lookup_measured_mp_s(model="RealESRGAN_x4plus", device="cuda", half=False)
    assert hit is not None
    assert hit["n"] == 1
    assert hit["mp_s"] > 0


def test_lookup_prefers_similar_workload(stats_dir: Path):
    # Tiny crop → high effective mp/s if mis-recorded; keep distinct rates.
    ss.record_speed_sample(
        model="M",
        device="cpu",
        seconds=1.0,
        work_mp=0.05,
        scale=2,
        src_w=64,
        src_h=64,
    )
    ss.record_speed_sample(
        model="M",
        device="cpu",
        seconds=20.0,
        work_mp=2.0,
        scale=4,
        src_w=512,
        src_h=512,
    )
    hit = ss.lookup_measured_mp_s(model="M", device="cpu", work_mp=2.0)
    assert hit is not None
    assert hit["mp_s"] == pytest.approx(0.1, rel=0.05)


def test_estimate_uses_calibrated_tighter_band(stats_dir: Path):
    cold_lo, cold_hi = estimate_time_seconds(
        256, 256, 4.0, device="cpu", measured_mp_s=0.2, calibrated=False
    )
    warm_lo, warm_hi = estimate_time_seconds(
        256, 256, 4.0, device="cpu", measured_mp_s=0.2, calibrated=True
    )
    assert warm_hi / warm_lo < cold_hi / cold_lo
    assert warm_lo < cold_lo  # warm load overhead is smaller


def test_estimate_api_reads_stats(client, stats_dir: Path):
    ss.record_from_job(
        model="RealESRGAN_x4plus",
        device="cpu",
        seconds=8.0,
        src_w=256,
        src_h=256,
        scale=2.0,
        half=False,
    )
    r = client.post(
        "/api/estimate",
        json={
            "src_w": 256,
            "src_h": 256,
            "scale": 2.0,
            "model": "RealESRGAN_x4plus",
            "device": "cpu",
            "model_ready": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["calibrated"] is True
    assert body["samples"] >= 1
    assert body["mp_s"] > 0
