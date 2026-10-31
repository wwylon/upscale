"""Load canonical help markdown from docs/help for API, UI, and MCP."""

from __future__ import annotations

import re
from pathlib import Path

HELP_TOPICS = [
    "size",
    "crop",
    "model",
    "filters",
    "output",
    "advanced",
    "preflight",
    "batch",
    "settings",
    "run",
    "cli",
    "mcp",
    "presets",
    "first_run",
    "shortcuts",
    "resume",
    "quality",
]

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")


def docs_root() -> Path:
    import os

    override = os.environ.get("UPSCALE_DOCS_DIR")
    if override:
        return Path(override) / "help"
    return Path(__file__).resolve().parent.parent / "docs" / "help"


def _require_safe(name: str, kind: str) -> str:
    if not name or not _SAFE_SEGMENT.match(name):
        raise ValueError(f"Invalid {kind}: {name!r}")
    return name


def _confined_file(root: Path, *rel_parts: str) -> Path:
    """Resolve file under root; raise if it escapes."""
    root_r = root.resolve()
    path = root_r.joinpath(*rel_parts).resolve()
    if not path.is_relative_to(root_r):
        raise ValueError(f"Path escapes docs root: {path}")
    return path


def topic_path(topic: str, lang: str = "en") -> Path:
    root = docs_root()
    topic = _require_safe(topic, "topic")
    if lang and lang != "en":
        lang = _require_safe(lang, "lang")
        candidate = _confined_file(root, lang, f"{topic}.md")
        if candidate.is_file():
            return candidate
    return _confined_file(root, f"{topic}.md")


def list_topics(lang: str = "en") -> list[dict[str, str]]:
    items = []
    for topic in HELP_TOPICS:
        try:
            path = topic_path(topic, lang)
        except ValueError:
            path = topic_path(topic, "en")
        title = topic.replace("_", " ").title()
        if path.is_file():
            first = path.read_text(encoding="utf-8").strip().splitlines()
            if first and first[0].startswith("# "):
                title = first[0][2:].strip()
        items.append({"id": topic, "title": title})
    return items


def get_help(topic: str, lang: str = "en") -> str:
    try:
        path = topic_path(topic, lang)
    except ValueError as exc:
        raise FileNotFoundError(f"Help topic not found: {topic}") from exc
    if not path.is_file():
        try:
            path = topic_path(topic, "en")
        except ValueError as exc:
            raise FileNotFoundError(f"Help topic not found: {topic}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Help topic not found: {topic}")
    return path.read_text(encoding="utf-8")
