"""Output size, print size, file-size and time estimate helpers."""

from __future__ import annotations

from typing import Literal

from .pipeline import compute_passes
from .schema import PrintSize, ResourceCheck, ResourceRow, ResourceStatus

VramFit = Literal["ok", "tight", "spill"]


# Rough megapixel/sec factors (calibrated later by benchmark).
SPEED_MP_S = {
    "cuda": {"RealESRGAN_x4plus": 2.5, "default": 3.0, "4x-spanx4-ch48": 8.0},
    "cpu": {"default": 0.15},
    "mps": {"default": 1.5},
}

BYTES_PER_PIXEL = {
    "png": (2.5, 4.0),
    "jpeg": (0.15, 0.4),
    "webp": (0.12, 0.35),
    "tiff": (3.0, 4.0),
    "bmp": (3.0, 3.0),
    "tga": (3.0, 4.0),
    "auto": (2.5, 4.0),
}


def output_dimensions(
    src_w: int,
    src_h: int,
    *,
    scale: float | None = None,
    target_size: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[int, int, float]:
    """Return (out_w, out_h, effective_scale)."""
    if width is not None and height is not None:
        eff = max(width / src_w, height / src_h)
        return width, height, eff
    if target_size is not None:
        long_edge = max(src_w, src_h)
        eff = target_size / long_edge
        return round(src_w * eff), round(src_h * eff), eff
    if scale is None:
        raise ValueError("Need scale, target_size, or width×height")
    return round(src_w * scale), round(src_h * scale), float(scale)


def megapixels(w: int, h: int) -> float:
    return (w * h) / 1_000_000


def print_sizes(w: int, h: int, dpis: tuple[int, ...] = (72, 150, 300, 600)) -> list[PrintSize]:
    out: list[PrintSize] = []
    for dpi in dpis:
        wi = w / dpi
        hi = h / dpi
        out.append(
            PrintSize(
                dpi=dpi,
                width_in=round(wi, 2),
                height_in=round(hi, 2),
                width_cm=round(wi * 2.54, 2),
                height_cm=round(hi * 2.54, 2),
            )
        )
    return out


def estimate_file_size_bytes(
    w: int,
    h: int,
    fmt: str = "auto",
) -> tuple[int, int]:
    """Return (low, high) estimated file size in bytes."""
    use = fmt
    if fmt == "auto" and max(w, h) > 20000:
        use = "tiff"
    elif fmt == "auto":
        use = "png"
    elif fmt == "jpeg_exif":
        use = "jpeg"
    elif fmt == "jp2":
        use = "jpeg"
    elif fmt in ("psd", "psb"):
        use = "tiff"
    elif fmt == "bmp":
        use = "bmp"
    elif fmt == "tga":
        use = "tga"
    lo, hi = BYTES_PER_PIXEL.get(use, BYTES_PER_PIXEL["png"])
    px = w * h
    return int(px * lo), int(px * hi)


def estimate_time_seconds(
    src_w: int,
    src_h: int,
    scale: float,
    *,
    model: str = "RealESRGAN_x4plus",
    device: str = "cuda",
    model_scale: int = 4,
    speed_table: dict | None = None,
    model_ready: bool = True,
    model_bytes: int | None = None,
    filter_factor: float = 1.0,
    measured_mp_s: float | None = None,
    model_load_seconds: float | None = None,
    algorithm: str | None = None,
    calibrated: bool = False,
) -> tuple[float, float]:
    """Return (low, high) seconds using MP/s factors (table or learned).

    Includes model-load overhead and optional download time when weights
    are not on disk yet. ``filter_factor`` multiplies compute time (from
    ``filters.estimate_time_factor``). When ``calibrated`` is true (stats from
    past runs on this machine), the range is tighter and load overhead is
    treated as warm-start.
    """
    from .speed_stats import work_megapixels

    table = speed_table or SPEED_MP_S
    dev = device if device in table else ("cuda" if device == "auto" else "cpu")
    factors = table.get(dev, table["cpu"])
    if algorithm:
        # Classical resize is typically much faster than neural SR.
        default_mp = 40.0 if dev == "cpu" else 120.0
        mp_s = measured_mp_s or factors.get(f"algo:{algorithm}", default_mp)
        work_mp = work_megapixels(
            src_w, src_h, scale, model_scale=model_scale, algorithm=algorithm
        )
        if work_mp <= 0:
            return 0.1, 0.5
        compute = (work_mp / max(mp_s, 1e-6)) * max(filter_factor, 1.0)
        load_s = 0.2 if calibrated else 0.5
        download_s = 0.0
    else:
        passes = compute_passes(scale, model_scale)
        if passes == 0:
            return 0.1, 0.5
        mp_s = measured_mp_s or factors.get(model, factors.get("default", 1.0))
        work_mp = work_megapixels(src_w, src_h, scale, model_scale=model_scale)
        compute = (work_mp / max(mp_s, 1e-6)) * max(filter_factor, 1.0)
        if model_load_seconds is not None:
            load_s = model_load_seconds
        elif calibrated:
            # Learned mp/s already includes some load amortization; keep a small warm bump.
            load_s = 2.0 if dev == "cpu" else 1.0
        else:
            load_s = 8.0 if dev == "cpu" else 4.0
            if model_bytes and model_bytes > 80_000_000:
                load_s *= 1.5
        download_s = 0.0
        if not model_ready:
            mb = (model_bytes or 67_000_000) / 1e6
            download_s = max(5.0, mb / 10.0)
    base = compute + load_s + download_s
    lo_f, hi_f = (0.85, 1.25) if calibrated else (0.7, 1.5)
    return round(base * lo_f, 1), round(base * hi_f, 1)


def model_fits_vram(
    model_bytes: int | None,
    free_vram_mb: int | None,
    *,
    tile: int = 512,
    fp16: bool = False,
) -> VramFit:
    """Classify whether model weights + one tile fit in free VRAM.

    ``ok`` — comfortable headroom · ``tight`` — may need smaller tile/fp16 ·
    ``spill`` — likely OOM / host spill; prefer smaller tile, CPU, or classical.
    """
    if free_vram_mb is None:
        return "ok"
    weights_mb = (model_bytes if model_bytes is not None else 67_000_000) / (1024 * 1024)
    bpp = 2 if fp16 else 4
    # ~8× activations vs raw RGB tile (conv features); weights inflate ~1.5× on device
    act_mb = (tile * tile * 3 * bpp * 8) / (1024 * 1024)
    need_mb = weights_mb * 1.5 + act_mb
    if free_vram_mb < need_mb * 0.55:
        return "spill"
    if free_vram_mb < need_mb:
        return "tight"
    return "ok"


def resource_check(
    *,
    out_w: int,
    out_h: int,
    tile: int,
    fp16: bool,
    model_bytes: int | None,
    vram_free_mb: int | None,
    ram_available_mb: int | None,
    disk_free_mb: int | None,
    model_ready: bool,
) -> ResourceCheck:
    rows: list[ResourceRow] = []
    blocked = False

    # Strip peak ≈ full-width × tile×scale float canvas + weight + uint8 result
    # (not 3× full float frames). Add a small spill reserve for mid-pass disk.
    strip_h = max(64, min(tile * 4, out_h))
    need_ram = (
        (out_w * strip_h * 3 * 4)  # float canvas
        + (out_w * strip_h * 4)  # weight map
        + (out_w * out_h * 3)  # uint8 result (often memmap)
        + (out_w * out_h * 3 * 0.25)  # spill / misc
    ) / (1024 * 1024)
    if ram_available_mb is None:
        rows.append(ResourceRow(name="RAM", status=ResourceStatus.ok, detail="unknown"))
    elif ram_available_mb < need_ram * 0.5:
        blocked = True
        rows.append(
            ResourceRow(
                name="RAM",
                status=ResourceStatus.not_enough,
                detail=f"~{need_ram:.0f} MB needed, {ram_available_mb} free",
                advice="Lower scale, enable disk intermediates, or close other apps",
            )
        )
    elif ram_available_mb < need_ram:
        rows.append(
            ResourceRow(
                name="RAM",
                status=ResourceStatus.tight,
                detail=f"~{need_ram:.0f} MB needed, {ram_available_mb} free",
                advice="Consider lower scale or smaller tile",
            )
        )
    else:
        rows.append(
            ResourceRow(
                name="RAM",
                status=ResourceStatus.ok,
                detail=f"~{need_ram:.0f} MB est., {ram_available_mb} free",
            )
        )

    # VRAM: weights + tile activations (model_fits_vram)
    fit = model_fits_vram(
        model_bytes, vram_free_mb, tile=tile, fp16=fp16
    )
    bpp = 2 if fp16 else 4
    tile_vram = (tile * tile * 3 * bpp * 8) / (1024 * 1024)
    weights_mb = (model_bytes or 0) / (1024 * 1024)
    detail = (
        f"tile {tile}, ~{tile_vram:.0f} MB act"
        + (f" + ~{weights_mb:.0f} MB weights" if weights_mb else "")
        + (f", {vram_free_mb} free" if vram_free_mb is not None else "")
    )
    if vram_free_mb is None:
        rows.append(ResourceRow(name="VRAM", status=ResourceStatus.ok, detail="CPU or unknown"))
    elif fit == "spill":
        blocked = True
        rows.append(
            ResourceRow(
                name="VRAM",
                status=ResourceStatus.not_enough,
                detail=detail,
                advice="Smaller tile, enable fp16, use CPU, or classical Method",
            )
        )
    elif fit == "tight":
        rows.append(
            ResourceRow(
                name="VRAM",
                status=ResourceStatus.tight,
                detail=detail,
                advice="Prefer smaller tile or fp16",
            )
        )
    else:
        rows.append(
            ResourceRow(
                name="VRAM",
                status=ResourceStatus.ok,
                detail=detail,
            )
        )

    out_mb = (out_w * out_h * 3) / (1024 * 1024)
    if disk_free_mb is None:
        rows.append(ResourceRow(name="Disk", status=ResourceStatus.ok, detail="unknown"))
    elif disk_free_mb < out_mb * 2:
        blocked = True
        rows.append(
            ResourceRow(
                name="Disk",
                status=ResourceStatus.not_enough,
                detail=f"need ~{out_mb * 2:.0f} MB free",
                advice="Free disk space or use JPEG/WebP",
            )
        )
    else:
        rows.append(
            ResourceRow(
                name="Disk",
                status=ResourceStatus.ok,
                detail=f"{disk_free_mb} MB free",
            )
        )

    if not model_ready:
        mb = (model_bytes or 0) / (1024 * 1024)
        rows.append(
            ResourceRow(
                name="Model",
                status=ResourceStatus.tight,
                detail=f"will download (~{mb:.0f} MB)" if mb else "will download",
                advice="Download before run or allow download-on-start",
            )
        )
    else:
        rows.append(ResourceRow(name="Model", status=ResourceStatus.ok, detail="ready"))

    return ResourceCheck(rows=rows, blocked=blocked)


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"
