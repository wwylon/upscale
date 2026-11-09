"""A0: help/locale path confinement."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from upscaler.help_content import get_help, topic_path
from upscaler.server import app


def test_topic_path_rejects_traversal():
    with pytest.raises(ValueError):
        topic_path("../secret")
    with pytest.raises(ValueError):
        topic_path("size", lang="../x")
    with pytest.raises(ValueError):
        topic_path("foo/bar")


def test_get_help_valid():
    body = get_help("filters")
    assert "Filter" in body or "filter" in body.lower()


def test_mcp_get_help_returns_markdown():
    """MCP get_help uses this import (`load_help`) and returns markdown."""
    from upscaler.mcp_server import load_help

    body = load_help("mcp", "en")
    assert isinstance(body, str)
    assert body.lstrip().startswith("#")


def test_api_help_traversal_404():
    client = TestClient(app)
    r = client.get("/api/help/..%2F..%2Fsecret")
    assert r.status_code == 404
    r2 = client.get("/api/help/filters", params={"lang": "../x"})
    assert r2.status_code == 404


def test_api_locale_traversal_404():
    client = TestClient(app)
    r = client.get("/api/locales/..%2F..%2Fetc")
    assert r.status_code == 404


def test_docs_dir_override_still_confined(tmp_path, monkeypatch):
    """UPSCALE_DOCS_DIR cannot escape topic/lang confinement."""
    import upscaler.help_content as hc

    docs = tmp_path / "docs"
    (docs / "help").mkdir(parents=True)
    # English topics live at help/{topic}.md (not help/en/).
    (docs / "help" / "filters.md").write_text("# Filters\n", encoding="utf-8")
    monkeypatch.setenv("UPSCALE_DOCS_DIR", str(docs))
    body = hc.get_help("filters", lang="en")
    assert "Filter" in body or "filter" in body.lower()
    with pytest.raises(ValueError):
        hc.topic_path("../secret", lang="en")
