"""Named target and option presets for CLI / API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TargetPreset:
    id: str
    label: str
    target_size: int
    description: str = ""


@dataclass(frozen=True)
class OptionPreset:
    id: str
    label: str
    settings: dict[str, Any]
    description: str = ""


TARGET_PRESETS: dict[str, TargetPreset] = {
    "fhd": TargetPreset("fhd", "Full HD long edge", 1920, "1080p-class width/height cap"),
    "4k": TargetPreset("4k", "4K UHD long edge", 3840, "Consumer 4K display target"),
    "print": TargetPreset("print", "Large print long edge", 6000, "Poster / print export"),
}

OPTION_PRESETS: dict[str, OptionPreset] = {
    "photo_4x": OptionPreset(
        "photo_4x",
        "Photo 4×",
        {
            "scale": 4.0,
            "model": "RealESRGAN_x4plus",
            "denoise": 0.4,
            "device": "auto",
        },
        "General photography at 4×",
    ),
    "art_4x": OptionPreset(
        "art_4x",
        "Art / anime 4×",
        {"scale": 4.0, "model": "RealESRGAN_x4plus_anime_6B", "device": "auto"},
    ),
    "fast_4x": OptionPreset(
        "fast_4x",
        "Fast 4× (SPAN)",
        {"scale": 4.0, "model": "4x-spanx4-ch48", "fast": True, "device": "auto"},
    ),
}


def list_presets() -> dict[str, Any]:
    return {
        "targets": [
            {"id": p.id, "label": p.label, "target_size": p.target_size, "description": p.description}
            for p in TARGET_PRESETS.values()
        ],
        "options": [
            {"id": p.id, "label": p.label, "settings": p.settings, "description": p.description}
            for p in OPTION_PRESETS.values()
        ],
    }


def resolve_target_preset(preset_id: str) -> TargetPreset:
    key = preset_id.lower()
    if key not in TARGET_PRESETS:
        raise KeyError(f"Unknown target preset: {preset_id}")
    return TARGET_PRESETS[key]


def resolve_option_preset(preset_id: str) -> OptionPreset:
    key = preset_id.lower()
    if key not in OPTION_PRESETS:
        raise KeyError(f"Unknown option preset: {preset_id}")
    return OPTION_PRESETS[key]
