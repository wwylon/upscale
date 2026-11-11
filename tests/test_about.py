"""About / license / update-check helpers."""

from __future__ import annotations

import json
from pathlib import Path

from upscale.about import (
    about_payload,
    check_for_updates,
    license_text,
    models_disk_summary,
    package_version,
)


def test_package_version():
    assert package_version()


def test_license_text_mentions_mit():
    text = license_text()
    assert "MIT" in text
    assert "Permission is hereby granted" in text


def test_about_payload_has_models_policy():
    a = about_payload()
    assert a["name"] == "Upscale"
    assert a["license"] == "MIT"
    assert "models_dir" in a
    assert "on demand" in a["models_policy"].lower() or "download" in a["models_policy"].lower()


def test_models_disk_summary_shape():
    s = models_disk_summary()
    assert "models_dir" in s
    assert "installed_count" in s
    assert isinstance(s["models"], list)


def test_check_updates_unconfigured():
    r = check_for_updates(releases_url="")
    assert r["status"] == "unconfigured"
    assert r["current"] == package_version()
    assert r["update_available"] is False


def test_check_updates_from_local_json(tmp_path: Path, monkeypatch):
    data = {"tag_name": "v99.0.0", "html_url": "https://example.com/r"}
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    # file:// may not work on all platforms via urlopen; use a tiny HTTP stub via data URL is hard —
    # instead monkeypatch urlopen.
    class _Resp:
        def read(self):
            return path.read_bytes()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=8.0):
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    r = check_for_updates(releases_url="https://example.com/releases/latest")
    assert r["status"] == "ok"
    assert r["update_available"] is True
    assert r["latest"] == "99.0.0"
