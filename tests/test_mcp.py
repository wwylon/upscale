from __future__ import annotations

import asyncio

import pytest

fastmcp = pytest.importorskip("mcp.server.fastmcp")

from upscaler.mcp_server import build_mcp, mcp_tool_catalog  # noqa: E402


def test_mcp_tool_catalog():
    cat = mcp_tool_catalog()
    assert "upscale_image" in cat["tools"]
    assert cat["transport_stdio"] == "upscale mcp"


def test_mcp_lists_tools():
    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert "get_help" in names
    assert "list_models" in names
    assert "compare_models" in names


def test_mcp_get_help_tool():
    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    help_tool = next(t for t in tools if t.name == "get_help")
    assert help_tool.description
    assert "topic" in help_tool.inputSchema.get("properties", {})
