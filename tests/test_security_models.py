"""Model name / download URL / naming confinement security tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from upscaler.models import validate_download_url, validate_model_name
from upscaler.naming import resolve_output_path
from upscaler.schema import NamePattern


def test_validate_model_name_rejects_traversal():
    with pytest.raises(ValueError):
        validate_model_name("../evil")
    with pytest.raises(ValueError):
        validate_model_name("a/b")
    with pytest.raises(ValueError):
        validate_model_name("a\\b")
    assert validate_model_name("RealESRGAN_x4plus") == "RealESRGAN_x4plus"


def test_validate_download_url_blocks_non_https_and_private():
    with pytest.raises(ValueError):
        validate_download_url("http://example.com/m.pth")
    with pytest.raises(ValueError):
        validate_download_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        validate_download_url("https://127.0.0.1/x.pth")
    with pytest.raises(ValueError):
        validate_download_url("https://evil.example/x.pth")
    ok = validate_download_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/x.pth"
    )
    assert ok.startswith("https://github.com/")


def test_api_rejects_evil_model_name():
    from fastapi.testclient import TestClient

    from upscaler.server import app

    client = TestClient(app)
    res = client.post(
        "/api/models",
        json={"name": "../evil", "url": "https://github.com/x/y.pth", "scale": 4},
    )
    assert res.status_code == 400


def test_name_pattern_stays_under_directory(tmp_path: Path):
    p = resolve_output_path(
        tmp_path,
        stem="../../escape",
        scale=4,
        model="m",
        ext="png",
        pattern=NamePattern(pattern="{stem}_{scale}x_{model}.{ext}"),
    )
    assert p is not None
    assert p.resolve().is_relative_to(tmp_path.resolve())
