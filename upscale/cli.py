"""CLI entry point for the upscale."""

from __future__ import annotations

import shlex
from pathlib import Path

import click

from .config import UpscaleConfig
from .engine import execute
from .models import DEFAULT_WEIGHTS_DIR, MODELS, add_external_model, all_models, model_status
from .naming import resolve_output_path
from .pipeline import DEFAULT_DISK_INTERMEDIATE_THRESHOLD
from .schema import FilterSettings, NamePattern, OnExists

IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.webp", "*.bmp")


def settings_to_cli(cfg: UpscaleConfig) -> str:
    """Build an equivalent CLI command string for the UI."""
    parts = ["upscale", "upscale", str(cfg.input or "INPUT"), str(cfg.output or "OUTPUT")]
    if cfg.scale is not None:
        parts += ["--scale", str(cfg.scale)]
    if cfg.target_size is not None:
        parts += ["--target-size", str(cfg.target_size)]
    if cfg.algorithm:
        parts += ["--algorithm", cfg.algorithm]
    else:
        parts += ["--model", cfg.model]
    if cfg.crop:
        x, y, w, h = cfg.crop
        parts += ["--crop", f"{x},{y},{w},{h}"]
    if cfg.strength != 1.0:
        parts += ["--strength", str(cfg.strength)]
    if cfg.denoise != 0.5:
        parts += ["--denoise", str(cfg.denoise)]
    if cfg.blend_model:
        parts += ["--blend-model", cfg.blend_model, "--blend", str(cfg.blend)]
    if cfg.format != "auto":
        parts += ["--format", cfg.format]
    if cfg.quality != 90:
        parts += ["--quality", str(cfg.quality)]
    f = cfg.filters or FilterSettings()
    if f.pre_denoise:
        parts += ["--pre-denoise", str(f.pre_denoise)]
    if f.jpeg_clean:
        parts += ["--jpeg-clean", str(f.jpeg_clean)]
    if f.deblur:
        parts += ["--deblur", str(f.deblur)]
    if f.edge:
        parts += ["--edge", str(f.edge)]
    if f.grain:
        parts += ["--grain", str(f.grain)]
    if f.color_contrast:
        parts += ["--color-contrast", str(f.color_contrast)]
    if f.halo_reduce:
        parts += ["--halo-reduce", str(f.halo_reduce)]
    if cfg.half:
        parts.append("--half")
    if cfg.cpu:
        parts.append("--cpu")
    if cfg.fast:
        parts.append("--fast")
    if cfg.sr_alpha:
        parts.append("--sr-alpha")
    if cfg.tile != "auto":
        parts += ["--tile", str(cfg.tile)]
    if cfg.overlap != 32:
        parts += ["--overlap", str(cfg.overlap)]
    if cfg.tiff_compression and cfg.tiff_compression != "lzw":
        parts += ["--tiff-compression", cfg.tiff_compression]
    if cfg.device not in ("auto", None) and not cfg.cpu:
        parts += ["--device", str(cfg.device)]
    if f.seamless:
        parts.append("--seamless")
    v = cfg.vectorize
    if v is not None and (
        cfg.algorithm == "vectorize_scale"
        or v.engine != "vtracer"
        or v.color_precision != 6
        or v.filter_speckle != 4
        or v.corner_threshold != 60
        or v.mode != "spline"
        or getattr(v, "max_trace_edge", 1024) != 1024
    ):
        if v.engine != "vtracer":
            parts += ["--vectorize-engine", v.engine]
        if v.color_precision != 6:
            parts += ["--vectorize-color-precision", str(v.color_precision)]
        if v.filter_speckle != 4:
            parts += ["--vectorize-filter-speckle", str(v.filter_speckle)]
        if v.corner_threshold != 60:
            parts += ["--vectorize-corner-threshold", str(v.corner_threshold)]
        if v.mode != "spline":
            parts += ["--vectorize-mode", v.mode]
        mte = getattr(v, "max_trace_edge", 1024)
        if mte != 1024:
            parts += ["--vectorize-max-trace-edge", str(mte)]
    if cfg.name_pattern is not None:
        parts += ["--name-pattern", cfg.name_pattern.pattern]
        on_ex = getattr(cfg.name_pattern, "on_exists", None)
        if on_ex is not None and str(on_ex.value if hasattr(on_ex, "value") else on_ex) != "number":
            parts += ["--on-exists", str(on_ex.value if hasattr(on_ex, "value") else on_ex)]
    return " ".join(shlex.quote(part) for part in parts)


def _common_options(fn):
    opts = [
        click.option("--scale", "-s", type=float, default=None),
        click.option("--target-size", "-t", type=int, default=None),
        click.option("--model", "-m", default="RealESRGAN_x4plus", show_default=True),
        click.option(
            "--algorithm",
            default=None,
            type=str,
            help=(
                "Classical / plugin resize instead of a neural model. "
                "Built-in: nearest, bilinear, bicubic, lanczos2, lanczos3. "
                "Paper: texture_residual, vectorize_scale (needs upscale[vectorize]). "
                "Or any name registered via register_algorithm."
            ),
        ),
        click.option("--tile", default="auto", show_default=True),
        click.option("--overlap", default=32, show_default=True, type=int),
        click.option(
            "--device",
            default="auto",
            show_default=True,
            type=click.Choice(["auto", "cuda", "cpu", "mps"], case_sensitive=False),
        ),
        click.option("--cpu", is_flag=True, default=False),
        click.option("--half", is_flag=True, default=False),
        click.option("--weights-dir", type=click.Path(path_type=Path), default=None),
        click.option(
            "--tiff-compression",
            default="lzw",
            show_default=True,
            type=click.Choice(
                ["lzw", "jpeg", "none", "deflate", "zstd"], case_sensitive=False
            ),
        ),
        click.option("--quiet", "-q", is_flag=True, default=False),
        click.option("--fast", is_flag=True, default=False),
        click.option("--sr-alpha", is_flag=True, default=False),
        click.option(
            "--disk-intermediate-threshold",
            type=int,
            default=DEFAULT_DISK_INTERMEDIATE_THRESHOLD,
            show_default=True,
        ),
        click.option("--crop", default=None, help="Crop region x,y,w,h"),
        click.option("--strength", type=float, default=1.0, show_default=True),
        click.option("--denoise", type=float, default=0.5, show_default=True),
        click.option("--blend-model", default=None),
        click.option("--blend", type=float, default=0.5, show_default=True),
        click.option(
            "--vectorize-engine",
            type=click.Choice(["vtracer", "autotrace"], case_sensitive=False),
            default="vtracer",
            show_default=True,
            help="vectorize_scale: tracing engine (vtracer or PyAutoTrace)",
        ),
        click.option(
            "--vectorize-color-precision",
            type=click.IntRange(1, 8),
            default=6,
            show_default=True,
            help="vectorize_scale: color bit depth (lower = flatter)",
        ),
        click.option(
            "--vectorize-filter-speckle",
            type=click.IntRange(0, 64),
            default=4,
            show_default=True,
            help="vectorize_scale: drop speckles smaller than N px",
        ),
        click.option(
            "--vectorize-corner-threshold",
            type=click.IntRange(0, 180),
            default=60,
            show_default=True,
            help="vectorize_scale: corner sensitivity (higher = sharper)",
        ),
        click.option(
            "--vectorize-mode",
            type=click.Choice(["spline", "polygon"], case_sensitive=False),
            default="spline",
            show_default=True,
            help="vectorize_scale: path style",
        ),
        click.option(
            "--vectorize-max-trace-edge",
            type=click.IntRange(256, 4096),
            default=1024,
            show_default=True,
            help="vectorize_scale: long-edge cap before tracing",
        ),
        click.option(
            "--format",
            "fmt",
            default="auto",
            show_default=True,
            type=click.Choice(
                [
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
                ],
                case_sensitive=False,
            ),
        ),
        click.option(
            "--quality",
            type=click.IntRange(1, 100),
            default=90,
            show_default=True,
        ),
        click.option(
            "--name-pattern",
            default=None,
            help="Output name pattern (default: none for single upscale; "
            "batch defaults to {stem}_{scale}x_{model}.{ext})",
        ),
        click.option(
            "--on-exists",
            default="number",
            show_default=True,
            type=click.Choice(["number", "overwrite", "skip"], case_sensitive=False),
        ),
        click.option("--pre-denoise", type=click.FloatRange(0.0, 1.0), default=0.0),
        click.option("--jpeg-clean", type=click.FloatRange(0.0, 1.0), default=0.0),
        click.option("--deblur", type=click.FloatRange(0.0, 1.0), default=0.0),
        click.option("--edge", type=click.FloatRange(0.0, 1.0), default=0.0),
        click.option("--grain", type=click.FloatRange(0.0, 1.0), default=0.0),
        click.option("--color-contrast", type=click.FloatRange(0.0, 1.0), default=0.0),
        click.option("--halo-reduce", type=click.FloatRange(0.0, 1.0), default=0.0),
        click.option("--seamless", is_flag=True, default=False),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def _parse_crop(crop: str | None) -> tuple[int, int, int, int] | None:
    if not crop:
        return None
    parts = [int(x.strip()) for x in crop.split(",")]
    if len(parts) != 4:
        raise click.UsageError("--crop must be x,y,w,h")
    if parts[2] <= 0 or parts[3] <= 0 or parts[0] < 0 or parts[1] < 0:
        raise click.UsageError("--crop requires non-negative x,y and positive w,h")
    return parts[0], parts[1], parts[2], parts[3]


def _cfg_from_opts(input: Path | None, output: Path | None, **kw) -> UpscaleConfig:
    """Build UpscaleConfig after validating shared fields via JobSettings."""
    from .schema import CropRect, JobSettings, VectorizeSettings

    crop = _parse_crop(kw.pop("crop", None))
    fmt = kw.pop("fmt", "auto")
    name_pattern = kw.pop("name_pattern")
    on_exists = kw.pop("on_exists")
    filters = FilterSettings(
        pre_denoise=kw.pop("pre_denoise"),
        jpeg_clean=kw.pop("jpeg_clean"),
        deblur=kw.pop("deblur"),
        edge=kw.pop("edge"),
        grain=kw.pop("grain"),
        color_contrast=kw.pop("color_contrast"),
        halo_reduce=kw.pop("halo_reduce"),
        seamless=kw.pop("seamless"),
    )
    vectorize = VectorizeSettings(
        engine=str(kw.pop("vectorize_engine") or "vtracer").lower(),
        color_precision=kw.pop("vectorize_color_precision"),
        filter_speckle=kw.pop("vectorize_filter_speckle"),
        corner_threshold=kw.pop("vectorize_corner_threshold"),
        mode=str(kw.pop("vectorize_mode") or "spline").lower(),
        max_trace_edge=kw.pop("vectorize_max_trace_edge"),
    )
    pattern = None
    if name_pattern:
        pattern = NamePattern(pattern=name_pattern, on_exists=OnExists(on_exists))

    # Validate bounds / enums through the canonical pydantic model.
    try:
        JobSettings.model_validate(
            {
                "scale": kw.get("scale"),
                "target_size": kw.get("target_size"),
                "model": kw.get("model") or "RealESRGAN_x4plus",
                "algorithm": kw.get("algorithm"),
                "tile": kw.get("tile", "auto"),
                "overlap": kw.get("overlap", 32),
                "device": "cpu" if kw.get("cpu") else (kw.get("device") or "auto"),
                "half": bool(kw.get("half")),
                "format": fmt,
                "quality": kw.get("quality", 90),
                "tiff_compression": kw.get("tiff_compression", "lzw"),
                "crop": CropRect(x=crop[0], y=crop[1], w=crop[2], h=crop[3])
                if crop
                else None,
                "strength": kw.get("strength", 1.0),
                "denoise": kw.get("denoise", 0.5),
                "blend_model": kw.get("blend_model"),
                "blend": kw.get("blend", 0.5),
                "filters": filters,
                "vectorize": vectorize,
                "name_pattern": pattern,
                "sr_alpha": bool(kw.get("sr_alpha")),
                "fast": bool(kw.get("fast")),
                "quiet": bool(kw.get("quiet")),
            }
        )
    except Exception as exc:  # noqa: BLE001
        raise click.UsageError(str(exc)) from exc

    return UpscaleConfig(
        input=input,
        output=output,
        format=fmt,
        crop=crop,
        filters=filters,
        vectorize=vectorize,
        name_pattern=pattern,
        **kw,
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """AI image upscaler — Real-ESRGAN with feather-blended tiles."""


@cli.command("upscale")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@_common_options
def upscale_cmd(input: Path, output: Path, **kw) -> None:
    """Upscale a single image."""
    cfg = _cfg_from_opts(input, output, **kw)
    execute(cfg)


@cli.command("batch")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@_common_options
def batch_cmd(input_dir: Path, output_dir: Path, **kw) -> None:
    """Upscale all images in INPUT_DIR into OUTPUT_DIR (non-recursive)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Batch always uses a name pattern (explicit or default).
    if kw.get("name_pattern") is None:
        kw["name_pattern"] = "{stem}_{scale}x_{model}.{ext}"
    files: list[Path] = []
    for pattern in IMAGE_GLOBS:
        files.extend(sorted(input_dir.glob(pattern)))
        files.extend(sorted(input_dir.glob(pattern.upper())))
    seen: set[Path] = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    if not unique:
        raise click.UsageError(f"No images found in {input_dir}")

    errors: list[str] = []
    for src in unique:
        cfg = _cfg_from_opts(src, output_dir / src.name, **kw)
        # Apply naming pattern into output_dir
        if cfg.name_pattern:
            resolved = resolve_output_path(
                output_dir,
                stem=src.stem,
                scale=cfg.scale or "auto",
                model=cfg.model,
                ext=cfg.format if cfg.format != "auto" else src.suffix.lstrip(".") or "png",
                pattern=cfg.name_pattern,
            )
            if resolved is None:
                click.echo(f"Skip {src.name} (exists)")
                continue
            cfg.output = resolved
        click.echo(f"=== {src.name} ===")
        try:
            execute(cfg)
        except Exception as exc:  # noqa: BLE001 — one bad file must not abort the batch
            errors.append(f"{src.name}: {exc}")
            click.echo(f"Failed {src.name}: {exc}", err=True)
    if errors:
        click.echo(f"Batch finished with {len(errors)} failure(s).", err=True)
        raise click.exceptions.Exit(1)


@cli.command("compare")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--crop", required=True, help="x,y,w,h")
@click.option("--models", required=True, help="Comma-separated model names")
@click.option("--scale", "-s", type=float, default=4.0, show_default=True)
@click.option("--device", default="auto")
@click.option("--half", is_flag=True, default=False)
def compare_cmd(
    input: Path,
    output_dir: Path,
    crop: str,
    models: str,
    scale: float,
    device: str,
    half: bool,
) -> None:
    """Compare models on a crop region."""
    from .compare import compare_models
    from .schema import CropRect

    parts = [int(x) for x in crop.split(",")]
    rect = CropRect(x=parts[0], y=parts[1], w=parts[2], h=parts[3])
    names = [m.strip() for m in models.split(",") if m.strip()]
    results = compare_models(
        input, output_dir, models=names, crop=rect, scale=scale, device=device, half=half
    )
    for r in results:
        status = r["error"] or r["path"]
        click.echo(f"{r['model']}: {r['seconds']}s → {status}")


@cli.group("models")
def models_group() -> None:
    """Manage built-in and external models."""


@models_group.command("list")
@click.option("--weights-dir", type=click.Path(path_type=Path), default=None)
def models_list(weights_dir: Path | None) -> None:
    wdir = weights_dir or DEFAULT_WEIGHTS_DIR
    catalog = all_models(wdir)
    col_w = max(len(k) for k in catalog) + 2
    for name, meta in catalog.items():
        st = model_status(name, wdir)
        desc = meta.get("description", "")
        click.echo(f"  {name:<{col_w}} [{st}] {desc}")


@models_group.command("add")
@click.argument("name")
@click.argument("url_or_path")
@click.option("--scale", type=int, default=4, show_default=True)
@click.option("--sha256", default=None)
@click.option("--weights-dir", type=click.Path(path_type=Path), default=None)
def models_add(
    name: str,
    url_or_path: str,
    scale: int,
    sha256: str | None,
    weights_dir: Path | None,
) -> None:
    wdir = weights_dir or DEFAULT_WEIGHTS_DIR
    p = Path(url_or_path)
    if p.exists():
        add_external_model(name, path=str(p), scale=scale, sha256=sha256, weights_dir=wdir)
    else:
        add_external_model(name, url=url_or_path, scale=scale, sha256=sha256, weights_dir=wdir)
    click.echo(f"Added external model {name}")


@models_group.command("verify")
@click.argument("name")
@click.option("--weights-dir", type=click.Path(path_type=Path), default=None)
def models_verify(name: str, weights_dir: Path | None) -> None:
    wdir = weights_dir or DEFAULT_WEIGHTS_DIR
    click.echo(f"{name}: {model_status(name, wdir)}")


# Keep bare `upscale models` listing for backward compat
@cli.command("models-legacy", hidden=True)
def list_models_legacy() -> None:
    for name, meta in MODELS.items():
        click.echo(f"  {name}  {meta.get('description', '')}")


@cli.command("suggest")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--fast", is_flag=True, default=False)
def suggest_cmd(input: Path, fast: bool) -> None:
    """Suggest a model from image statistics."""
    import json

    from .suggest import suggest_model

    click.echo(json.dumps(suggest_model(input, fast=fast), indent=2))


@cli.group("presets")
def presets_group() -> None:
    """List target and option presets."""


@presets_group.command("list")
def presets_list_cmd() -> None:
    """Print FHD/4K/print targets and Photo 4× options."""
    import json

    from .presets import list_presets

    click.echo(json.dumps(list_presets(), indent=2))


@cli.command("benchmark")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--model", "-m", default="RealESRGAN_x4plus", show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--crop-size", default=64, show_default=True, type=int)
@click.option("--scale", "-s", default=2.0, show_default=True, type=float)
def benchmark_cmd(input: Path, model: str, device: str, crop_size: int, scale: float) -> None:
    """Run a tiny crop timing benchmark (saved under user config)."""
    import json

    from .benchmark import load_benchmark_history, run_benchmark

    entry = run_benchmark(
        input, model=model, device=device, crop_size=crop_size, scale=scale
    )
    click.echo(json.dumps({"last": entry, "history": load_benchmark_history()[-5:]}, indent=2))


@cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8787, show_default=True, type=int)
def serve_cmd(host: str, port: int) -> None:
    """Launch the FastAPI web UI and job API.

    Default bind is 127.0.0.1. Non-loopback hosts require UPSCALE_ALLOW_REMOTE=1
    because the job/model APIs are unauthenticated.
    """
    try:
        from .server import run_server
    except ImportError as exc:
        raise click.ClickException(
            "Server extras required: pip install 'upscale[server]'"
        ) from exc
    try:
        run_server(host=host, port=port)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("mcp")
def mcp_cmd() -> None:
    """Run the MCP stdio server for AI agents."""
    try:
        from .mcp_server import run_stdio
    except ImportError as exc:
        raise click.ClickException(
            "MCP extras required: pip install 'upscale[mcp]'"
        ) from exc
    run_stdio()


@cli.command("ui")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=7860, show_default=True, type=int)
@click.option("--share", is_flag=True, default=False)
def ui_cmd(host: str, port: int, share: bool) -> None:
    """Launch Gradio UI (legacy; prefer `upscale serve`)."""
    import ipaddress
    import os

    click.echo("Note: prefer `upscale serve` for the design UI.", err=True)
    allow_remote = os.environ.get("UPSCALE_ALLOW_REMOTE", "").strip() in (
        "1",
        "true",
        "yes",
    )
    loopback = False
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host in ("localhost",)
    if share and not allow_remote:
        raise click.ClickException(
            "Refusing --share without UPSCALE_ALLOW_REMOTE=1 "
            "(Gradio share exposes an unauthenticated UI)."
        )
    if not loopback and not allow_remote:
        raise click.ClickException(
            f"Refusing to bind non-loopback host {host!r}. "
            "Set UPSCALE_ALLOW_REMOTE=1 to allow."
        )
    try:
        from .web import launch
    except ImportError as exc:
        raise click.ClickException(
            "Gradio required: pip install 'upscale[ui]'"
        ) from exc
    launch(host=host, port=port, share=share)


main = upscale_cmd
