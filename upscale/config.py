"""Shared configuration for CLI and web UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import DEFAULT_WEIGHTS_DIR
from .pipeline import DEFAULT_DISK_INTERMEDIATE_THRESHOLD
from .schema import FilterSettings, NamePattern, VectorizeSettings


@dataclass
class UpscaleConfig:
    input: Path | None = None
    output: Path | None = None
    scale: float | None = None
    target_size: int | None = None
    width: int | None = None
    height: int | None = None
    model: str = "RealESRGAN_x4plus"
    algorithm: str | None = None
    tile: str | int = "auto"
    overlap: int = 32
    device: str = "auto"
    cpu: bool = False
    half: bool = False
    weights_dir: Path | None = None
    tiff_compression: str = "lzw"
    quiet: bool = False
    fast: bool = False
    sr_alpha: bool = False
    disk_intermediate_threshold: int = DEFAULT_DISK_INTERMEDIATE_THRESHOLD
    # Phase 1+
    crop: tuple[int, int, int, int] | None = None
    strength: float = 1.0
    denoise: float = 0.5
    blend_model: str | None = None
    blend: float = 0.5
    format: str = "auto"
    quality: int = 90
    image_meta: object | None = None  # io_utils.ImageMeta, set by engine on load
    filters: FilterSettings = field(default_factory=FilterSettings)
    vectorize: VectorizeSettings = field(default_factory=VectorizeSettings)
    name_pattern: NamePattern | None = None
    job_dir: Path | None = None
    resume_from_checkpoint: bool = False
    mid_pass_previews: bool = True
    cancel_event: object | None = None
    pause_event: object | None = None

    def resolved_weights_dir(self) -> Path:
        return self.weights_dir or DEFAULT_WEIGHTS_DIR

    def resolved_device_str(self) -> str:
        return "cpu" if self.cpu else self.device
