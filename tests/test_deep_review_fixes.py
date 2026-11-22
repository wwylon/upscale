"""Regression tests for the 2026-11-19 deep-review wave."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from png_fixtures import TINY_PNG

from upscale.checkpoint import list_checkpoints
from upscale.compare import build_variants
from upscale.job_prep import MAX_OUTPUT_PIXELS
from upscale.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_list_checkpoints_corrupt_json(tmp_path: Path):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "strip.json").write_text("{not-json", encoding="utf-8")
    assert list_checkpoints(tmp_path) == []


def test_batch_algorithm_naming_token(tmp_path: Path):
    """When --algorithm is set, name patterns should use the algorithm token."""
    from upscale.config import UpscaleConfig
    from upscale.naming import resolve_output_path

    cfg = UpscaleConfig(algorithm="lanczos3", model="RealESRGAN_x4plus", scale=2.0)
    model_token = cfg.algorithm if cfg.algorithm else cfg.model
    out = resolve_output_path(
        tmp_path,
        stem="a",
        scale=2,
        model=model_token,
        ext="png",
    )
    assert out is not None
    assert "lanczos3" in out.name
    assert "RealESRGAN" not in out.name


def test_filter_variants_use_classical():
    variants = build_variants(filters=["edge"], base_model="RealESRGAN_x4plus")
    assert len(variants) == 1
    assert variants[0]["kind"] == "filter"
    assert variants[0]["algorithm"] == "lanczos3"


def test_compare_rejects_svg(client: TestClient):
    r = client.post(
        "/api/compare",
        data={
            "body": json.dumps(
                {
                    "models": ["RealESRGAN_x4plus"],
                    "crop": {"x": 0, "y": 0, "w": 8, "h": 8},
                    "scale": 2,
                }
            )
        },
        files={"file": ("icon.svg", b"<svg/>", "image/svg+xml")},
    )
    assert r.status_code == 400


def test_suggest_rejects_svg(client: TestClient):
    r = client.post(
        "/api/suggest",
        files={"file": ("icon.svg", b"<svg/>", "image/svg+xml")},
    )
    assert r.status_code == 400


def test_suggest_accepts_png(client: TestClient):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(
        buf, format="PNG"
    )
    r = client.post(
        "/api/suggest",
        files={"file": ("sample.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "model" in body
    assert "kind" in body


def test_create_job_scale_megapixel_reject(client: TestClient):
    huge_scale = (MAX_OUTPUT_PIXELS // 64) + 10
    r = client.post(
        "/api/jobs",
        data={"settings": json.dumps({"scale": float(huge_scale)})},
        files={"file": ("tiny.png", TINY_PNG, "image/png")},
    )
    assert r.status_code == 400
    assert "pixel" in r.text.lower() or "exceed" in r.text.lower() or "budget" in r.text.lower()


def test_api_estimate_still_works(client: TestClient):
    r = client.post(
        "/api/estimate",
        json={"src_w": 64, "src_h": 64, "scale": 4.0, "model": "RealESRGAN_x4plus"},
    )
    assert r.status_code == 200
    assert "seconds" in r.json()


def test_api_preflight(client: TestClient):
    r = client.post(
        "/api/preflight",
        json={
            "src_w": 64,
            "src_h": 64,
            "scale": 4.0,
            "model": "RealESRGAN_x4plus",
            "device": "cpu",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body
    assert "checks" in body
    assert "target_w" in body


def test_alpha_policy_lanczos():
    from upscale.alpha_policy import upsample_alpha

    a = np.full((8, 8), 200, dtype=np.uint8)
    out = upsample_alpha(a, scale=2.0, target_w=16, target_h=16, sr_alpha=False)
    assert out.shape == (16, 16)
    assert out.dtype == np.uint8


def test_help_backends_topic(client: TestClient):
    r = client.get("/api/help/backends")
    assert r.status_code == 200
    md = r.json()["markdown"]
    assert "ONNX" in md or "Torch" in md


def test_memmap_util_spill(tmp_path: Path):
    from upscale.memmap_util import is_memmap, materialize, spill_uint8_rows

    arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
    dest = tmp_path / "x.bin"
    spill_uint8_rows(arr, dest, chunk=2)
    assert dest.stat().st_size == 12
    assert not is_memmap(arr)
    assert materialize(arr).shape == arr.shape
