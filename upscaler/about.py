"""About, license, and update-check helpers for UI / desktop."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Any

from .models import DEFAULT_WEIGHTS_DIR, all_models, model_status

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LICENSE_PATH = PROJECT_ROOT / "LICENSE"

# Optional: GitHub Releases API URL or a static JSON with tag_name / html_url
DEFAULT_RELEASES_URL = os.environ.get("UPSCALE_RELEASES_URL", "").strip()


def package_version() -> str:
    try:
        return metadata.version("upscaler")
    except metadata.PackageNotFoundError:
        from . import __version__

        return __version__


def license_text() -> str:
    if LICENSE_PATH.is_file():
        return LICENSE_PATH.read_text(encoding="utf-8")
    return (
        "MIT License\n\n"
        "Copyright (c) 2026 Upscale contributors\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy "
        "of this software..."
    )


def about_payload() -> dict[str, Any]:
    models_dir = Path(os.environ.get("UPSCALE_MODELS_DIR", DEFAULT_WEIGHTS_DIR))
    return {
        "name": "Upscale",
        "version": package_version(),
        "license": "MIT",
        "description": (
            "AI image upscaler — chained Real-ESRGAN / HAT / SPAN with "
            "feather-blended tiles. Classical resize via OpenCV when available."
        ),
        "models_dir": str(models_dir),
        "models_policy": (
            "Model weights are downloaded on demand into the models folder "
            "(desktop: %LOCALAPPDATA%\\Upscale\\models). They are not baked into "
            "the slim installer. Optional tiny SPAN may ship later for offline first run."
        ),
    }


def models_disk_summary(weights_dir: Path | None = None) -> dict[str, Any]:
    wdir = weights_dir or Path(os.environ.get("UPSCALE_MODELS_DIR", DEFAULT_WEIGHTS_DIR))
    catalog = all_models(wdir)
    installed: list[dict[str, Any]] = []
    total = 0
    for name, meta in sorted(catalog.items()):
        st = model_status(name, wdir)
        if st != "installed":
            continue
        url_path = (meta.get("url") or "").split("?")[0]
        ext = Path(url_path).suffix or ".pth"
        dest = Path(meta["path"]) if meta.get("path") else wdir / f"{name}{ext}"
        size = dest.stat().st_size if dest.is_file() else int(meta.get("size_bytes") or 0)
        total += size
        installed.append({"name": name, "path": str(dest), "size_bytes": size})
    return {
        "models_dir": str(wdir),
        "installed_count": len(installed),
        "total_bytes": total,
        "models": installed,
    }


def _parse_version(tag: str) -> tuple[int, ...]:
    s = tag.strip().lstrip("vV")
    parts: list[int] = []
    for p in s.split("."):
        num = ""
        for ch in p:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts) or (0,)


def check_for_updates(
    *,
    releases_url: str | None = None,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Compare local version to latest release tag.

    ``releases_url`` may be a GitHub Releases API endpoint
    (``…/repos/OWNER/REPO/releases/latest``) or any JSON with ``tag_name``.
    """
    current = package_version()
    url = (releases_url if releases_url is not None else DEFAULT_RELEASES_URL).strip()
    if not url:
        return {
            "status": "unconfigured",
            "current": current,
            "latest": None,
            "update_available": False,
            "message": "Set UPSCALE_RELEASES_URL to enable update checks.",
            "url": None,
        }
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "upscale"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        if isinstance(data, list) and data:
            data = data[0]
        tag = str(data.get("tag_name") or data.get("version") or "")
        html = data.get("html_url") or data.get("url")
        if not tag:
            return {
                "status": "error",
                "current": current,
                "latest": None,
                "update_available": False,
                "message": "Release JSON missing tag_name",
                "url": html,
            }
        newer = _parse_version(tag) > _parse_version(current)
        return {
            "status": "ok",
            "current": current,
            "latest": tag.lstrip("vV"),
            "update_available": newer,
            "message": (
                f"Update available: {tag}" if newer else f"Up to date ({current})"
            ),
            "url": html,
        }
    except urllib.error.URLError as exc:
        return {
            "status": "error",
            "current": current,
            "latest": None,
            "update_available": False,
            "message": f"Could not reach releases: {exc}",
            "url": None,
        }
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "status": "error",
            "current": current,
            "latest": None,
            "update_available": False,
            "message": f"Bad releases response: {exc}",
            "url": None,
        }
