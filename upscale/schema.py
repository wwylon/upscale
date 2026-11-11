"""Shared job settings and event types for CLI, REST, MCP, and UIs."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class OnExists(str, Enum):
    number = "number"
    overwrite = "overwrite"
    skip = "skip"


class ResourceStatus(str, Enum):
    ok = "OK"
    tight = "TIGHT"
    not_enough = "NOT_ENOUGH"


class NamePattern(BaseModel):
    """Output filename pattern with collision policy."""

    pattern: str = "{stem}_{scale}x_{model}.{ext}"
    on_exists: OnExists = OnExists.number

    def render(
        self,
        *,
        stem: str,
        scale: float | str,
        model: str,
        ext: str,
        w: int | None = None,
        h: int | None = None,
        date: str | None = None,
        n: int | None = None,
    ) -> str:
        from datetime import datetime, timezone

        scale_s = f"{float(scale):g}" if isinstance(scale, (int, float)) else str(scale)
        ext_clean = ext.lstrip(".")
        model_safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)
        def _safe_token(s: str) -> str:
            # Basename only — strip path separators and parent refs.
            s = s.replace("\\", "/").split("/")[-1]
            s = s.replace("..", "_")
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

        values = {
            "stem": _safe_token(stem),
            "scale": scale_s,
            "model": model_safe,
            "ext": _safe_token(ext_clean),
            "w": "" if w is None else str(w),
            "h": "" if h is None else str(h),
            "date": date or datetime.now(timezone.utc).strftime("%Y%m%d"),
            "n": "" if n is None else str(n),
        }
        out = self.pattern
        for key, val in values.items():
            out = out.replace("{" + key + "}", val)
        # Final path segment only (reject pattern smuggling of dirs).
        out = out.replace("\\", "/").split("/")[-1]
        if not out or out in (".", ".."):
            out = "output.png"
        return out


class CropRect(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class FilterSettings(BaseModel):
    pre_denoise: float = Field(default=0.0, ge=0.0, le=1.0)
    jpeg_clean: float = Field(default=0.0, ge=0.0, le=1.0)
    deblur: float = Field(default=0.0, ge=0.0, le=1.0)
    edge: float = Field(default=0.0, ge=0.0, le=1.0)
    grain: float = Field(default=0.0, ge=0.0, le=1.0)
    color_contrast: float = Field(default=0.0, ge=0.0, le=1.0)
    halo_reduce: float = Field(default=0.0, ge=0.0, le=1.0)
    seamless: bool = False
    # Face restore reserved; engine ignores until a restorer model is wired.
    face_restore: float = Field(default=0.0, ge=0.0, le=1.0)


class VectorizeSettings(BaseModel):
    """Trace knobs for ``algorithm=vectorize_scale`` (VTracer or PyAutoTrace)."""

    engine: Literal["vtracer", "autotrace"] = Field(
        default="vtracer",
        description="vtracer = Visioncortex; autotrace = classic AutoTrace (PyAutoTrace).",
    )
    color_precision: int = Field(
        default=6, ge=1, le=8, description="Fewer bits → flatter logo-like colors."
    )
    filter_speckle: int = Field(
        default=4, ge=0, le=64, description="Drop speckles smaller than this many pixels."
    )
    corner_threshold: int = Field(
        default=60, ge=0, le=180, description="Higher → more corners / sharper angles."
    )
    mode: Literal["spline", "polygon"] = Field(
        default="spline",
        description="VTracer only: spline = smooth curves; polygon = hard edges.",
    )
    max_trace_edge: int = Field(
        default=1024,
        ge=256,
        le=4096,
        description="Long-edge cap before tracing (higher = more detail, slower).",
    )


class JobSettings(BaseModel):
    """Canonical settings shared by CLI, web, desktop, MCP, Android."""

    scale: float | None = None
    target_size: int | None = None
    width: int | None = None
    height: int | None = None
    model: str = "RealESRGAN_x4plus"
    # When set, neural model is skipped (classical / plugin resize).
    # Built-ins or registered plugin names (validated at execute time).
    algorithm: str | None = None
    vectorize: VectorizeSettings = Field(default_factory=lambda: VectorizeSettings())
    tile: str | int = "auto"
    overlap: int = Field(32, ge=0, le=512)

    @field_validator("tile")
    @classmethod
    def _tile_ok(cls, v: str | int) -> str | int:
        if isinstance(v, str):
            if v.lower() == "auto":
                return "auto"
            if v.isdigit():
                v = int(v)
            else:
                raise ValueError("tile must be 'auto' or an integer")
        if isinstance(v, int) and not (32 <= v <= 2048):
            raise ValueError("tile must be between 32 and 2048")
        return v
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    half: bool = False
    format: Literal[
        "auto",
        "png",
        "jpeg",
        "jpeg_exif",
        "webp",
        "tiff",
        "bmp",
        "tga",
        "jp2",
        "psd",
        "psb",
    ] = "auto"
    quality: int = Field(90, ge=1, le=100)
    tiff_compression: Literal["lzw", "jpeg", "none", "deflate", "zstd"] = "lzw"
    crop: CropRect | None = None
    strength: float = Field(1.0, ge=0.0, le=1.0)
    denoise: float = Field(0.5, ge=0.0, le=1.0)
    blend_model: str | None = None
    blend: float = Field(0.5, ge=0.0, le=1.0)
    filters: FilterSettings = Field(default_factory=lambda: FilterSettings())
    # None = write exactly to the requested output path (CLI OUTPUT / API result.*).
    # Set explicitly for batch-style patterned names.
    name_pattern: NamePattern | None = None
    sr_alpha: bool = False
    disk_intermediate_threshold: int | None = None
    fast: bool = False
    quiet: bool = False
    weights_dir: str | None = None
    job_dir: str | None = None
    mid_pass_previews: bool = True

    @model_validator(mode="after")
    def _scale_xor_target(self) -> JobSettings:
        if self.scale is not None and self.target_size is not None:
            raise ValueError("Specify scale or target_size, not both")
        return self

    def to_recipe_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_recipe_file(cls, path: Path | str) -> JobSettings:
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save_recipe(self, path: Path | str) -> Path:
        import json

        p = Path(path)
        p.write_text(json.dumps(self.to_recipe_dict(), indent=2), encoding="utf-8")
        return p


class ProgressEvent(BaseModel):
    type: Literal[
        "progress",
        "pass_complete",
        "tile",
        "done",
        "error",
        "paused",
        "resumed",
        "cancelled",
        "queued",
        "loading_model",
        "pre_filters",
        "filters",
        "writing",
    ] = "progress"
    pass_index: int | None = None  # alias; wire format also uses "pass"
    pass_: int | None = Field(None, alias="pass")
    passes: int | None = None
    tile: int | None = None
    tiles: int | None = None
    pct: float = 0.0
    eta_s: float | None = None
    message: str = ""
    artifact: str | None = None

    model_config = {"populate_by_name": True}


class PassArtifact(BaseModel):
    pass_index: int
    path: str
    width: int
    height: int


class ResourceRow(BaseModel):
    name: str
    status: ResourceStatus
    detail: str = ""
    advice: str = ""


class ResourceCheck(BaseModel):
    rows: list[ResourceRow] = Field(default_factory=list)
    blocked: bool = False

    @property
    def ok(self) -> bool:
        return not self.blocked


class SystemInfo(BaseModel):
    device: str
    cuda_available: bool = False
    mps_available: bool = False
    gpu_name: str | None = None
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    ram_total_mb: int | None = None
    ram_available_mb: int | None = None
    disk_free_mb: int | None = None
    auto_tile: int | None = None
    models_dir: str | None = None
    backend: str = "torch"


class ModelInfo(BaseModel):
    name: str
    description: str = ""
    scale: int = 4
    source: Literal["built-in", "external", "community"] = "built-in"
    status: Literal["installed", "not_downloaded", "failed"] = "not_downloaded"
    path: str | None = None
    sha256: str | None = None
    url: str | None = None
    license: str | None = None
    tags: list[str] = Field(default_factory=list)
    size_bytes: int | None = None


class PrintSize(BaseModel):
    dpi: int
    width_in: float
    height_in: float
    width_cm: float
    height_cm: float


LANGUAGE_CATALOG: list[dict[str, Any]] = [
    {"code": "en", "name": "English", "enabled": True},
    {"code": "pl", "name": "Polski", "enabled": True},
    {"code": "de", "name": "Deutsch", "enabled": False},
    {"code": "fr", "name": "Français", "enabled": False},
    {"code": "es", "name": "Español", "enabled": False},
    {"code": "it", "name": "Italiano", "enabled": False},
    {"code": "pt", "name": "Português", "enabled": False},
    {"code": "nl", "name": "Nederlands", "enabled": False},
    {"code": "ja", "name": "日本語", "enabled": False},
    {"code": "zh", "name": "中文", "enabled": True},
    {"code": "ar", "name": "العربية", "enabled": True},
    {"code": "ko", "name": "한국어", "enabled": False},
    {"code": "uk", "name": "Українська", "enabled": False},
    {"code": "cs", "name": "Čeština", "enabled": False},
    {"code": "sv", "name": "Svenska", "enabled": False},
]
