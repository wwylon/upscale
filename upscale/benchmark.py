"""Tiny-crop inference timing saved to user config."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .config import UpscaleConfig
from .engine import execute


def user_config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        root = Path(base) / "upscale"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "upscale"
    root.mkdir(parents=True, exist_ok=True)
    return root


def benchmark_path() -> Path:
    return user_config_dir() / "benchmark.json"


def load_benchmark_history() -> list[dict[str, Any]]:
    path = benchmark_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return data
    return data.get("runs", [])


def save_benchmark_entry(entry: dict[str, Any]) -> None:
    history = load_benchmark_history()
    history.append(entry)
    history = history[-50:]
    benchmark_path().write_text(json.dumps({"runs": history}, indent=2), encoding="utf-8")


def run_benchmark(
    image: Path | str,
    *,
    model: str = "RealESRGAN_x4plus",
    device: str = "cpu",
    crop_size: int = 64,
    scale: float = 2.0,
) -> dict[str, Any]:
    """
    Time a small centered crop upscale (mock-friendly when execute is patched).
    """
    from PIL import Image

    src = Path(image)
    with Image.open(src) as im:
        w, h = im.size
    cs = min(crop_size, w, h)
    x = max(0, (w - cs) // 2)
    y = max(0, (h - cs) // 2)
    out = user_config_dir() / f"bench_{int(time.time())}.png"
    cfg = UpscaleConfig(
        input=src,
        output=out,
        scale=scale,
        model=model,
        device=device,
        cpu=device == "cpu",
        quiet=True,
        crop=(x, y, cs, cs),
    )
    t0 = time.perf_counter()
    try:
        result = execute(cfg)
        seconds = round(time.perf_counter() - t0, 3)
        entry = {
            "model": model,
            "device": device,
            "crop_size": cs,
            "scale": scale,
            "seconds": seconds,
            "output": str(result),
            "image": str(src),
        }
        save_benchmark_entry(entry)
        return entry
    finally:
        if out.is_file():
            out.unlink(missing_ok=True)
