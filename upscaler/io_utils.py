"""Image I/O: input normalization, alpha handling, output format selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Auto format switches from PNG to TIFF when the long edge exceeds this (px).
LONG_EDGE_TIFF_THRESHOLD = 20_000

# Formats the product accepts for JobSettings / CLI
OUTPUT_FORMATS = (
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
)

# Input extensions accepted by CLI / web (browsers cannot preview all of these).
# Raster only — vector formats (SVG, EPS, …) are rejected.
INPUT_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
        ".bmp",
        ".tga",
        ".psd",
        ".psb",
        ".jp2",
        ".j2k",
        ".jpf",
        ".jpx",
    }
)

# Explicitly blocked even if a browser reports image/* for them.
VECTOR_SUFFIXES = frozenset(
    {
        ".svg",
        ".svgz",
        ".eps",
        ".ai",
        ".emf",
        ".wmf",
    }
)
VECTOR_MIME_PREFIXES = (
    "image/svg",
    "image/svg+xml",
    "application/postscript",
    "application/illustrator",
)

TIFF_COMPRESSIONS = ("lzw", "jpeg", "none", "deflate", "zstd")


def is_vector_input(filename: str | None = None, content_type: str | None = None) -> bool:
    """True for SVG / EPS / AI / EMF / WMF (not supported by the raster pipeline)."""
    suffix = Path(filename or "").suffix.lower()
    if suffix in VECTOR_SUFFIXES:
        return True
    ctype = (content_type or "").split(";")[0].strip().lower()
    if not ctype:
        return False
    return any(ctype == p or ctype.startswith(p + "+") for p in VECTOR_MIME_PREFIXES) or ctype in VECTOR_MIME_PREFIXES


def require_raster_suffix(filename: str | None) -> str:
    """Return lowercased suffix if allowed; raise ValueError for vector / unknown."""
    suffix = (Path(filename or "input.bin").suffix or ".bin").lower()
    if suffix in VECTOR_SUFFIXES or is_vector_input(filename):
        raise ValueError(
            f"Vector images are not supported ({suffix or 'unknown'}). "
            "Rasterize to PNG, TIFF, or another supported bitmap format first."
        )
    if suffix not in INPUT_SUFFIXES:
        raise ValueError(
            f"Unsupported input type {suffix}. "
            f"Allowed: {', '.join(sorted(INPUT_SUFFIXES))}"
        )
    return suffix


@dataclass
class ImageMeta:
    """Optional sidecar metadata preserved across the pipeline."""

    exif: bytes | None = None
    icc_profile: bytes | None = None
    info: dict[str, Any] = field(default_factory=dict)


def load_rgba(path: Path) -> tuple[np.ndarray, np.ndarray | None, str]:
    """
    Load image once as uint8 RGB (plus optional alpha).
    Returns (rgb, alpha_or_none, original_mode).
    """
    rgb, alpha, mode, _meta = load_rgba_with_meta(path)
    return rgb, alpha, mode


def _strip_orientation_exif(exif_bytes: bytes | None) -> bytes | None:
    """Drop Orientation (tag 0x0112) after we have transposed the pixels."""
    if not exif_bytes:
        return None
    try:
        from PIL import Image

        im = Image.new("RGB", (1, 1))
        im.info["exif"] = exif_bytes
        exif = im.getexif()
        if 0x0112 in exif:
            del exif[0x0112]
        data = exif.tobytes()
        return data or None
    except Exception:  # noqa: BLE001
        return exif_bytes


def _fill_transparent_rgb(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Replace RGB under fully transparent pixels with nearest opaque colour."""
    if alpha is None or rgb is None:
        return rgb
    mask = alpha == 0
    if not np.any(mask) or np.all(mask):
        return rgb
    opaque = ~mask
    # Cheap fill: use mean of opaque pixels (avoids dark SR fringes on A=0).
    fill = rgb[opaque].mean(axis=0).astype(np.uint8)
    out = rgb.copy()
    out[mask] = fill
    return out


def _pil_to_rgba(
    img: Image.Image, *, meta: ImageMeta | None = None
) -> tuple[np.ndarray, np.ndarray | None, str, ImageMeta]:
    from PIL import ImageOps

    original_mode = img.mode
    meta = meta or ImageMeta()
    try:
        raw_exif = img.info.get("exif")
        if raw_exif:
            meta.exif = bytes(raw_exif)
        elif hasattr(img, "getexif"):
            exif_obj = img.getexif()
            if exif_obj:
                meta.exif = exif_obj.tobytes()
    except Exception:  # noqa: BLE001
        pass
    # Apply EXIF orientation to pixels, then drop the tag so saves stay upright.
    try:
        img = ImageOps.exif_transpose(img)
        meta.exif = _strip_orientation_exif(meta.exif)
    except Exception:  # noqa: BLE001
        pass
    icc = img.info.get("icc_profile")
    if icc:
        meta.icc_profile = bytes(icc)
    meta.info = {k: v for k, v in img.info.items() if k not in ("exif", "icc_profile")}

    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        original_mode = img.mode

    alpha: np.ndarray | None = None
    if img.mode in ("RGBA", "LA"):
        alpha = np.array(img.split()[-1], dtype=np.uint8)
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        rgb = _fill_transparent_rgb(rgb, alpha)
        return rgb, alpha, "RGBA" if img.mode == "RGBA" else original_mode, meta

    if img.mode != "RGB":
        img = img.convert("RGB")

    return np.array(img, dtype=np.uint8), None, original_mode, meta


def _load_via_tifffile(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None, str, ImageMeta]:
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        planar = getattr(page, "planarconfig", None)
        # tifffile: 1=chunky/contiguous, 2=planar. Fall back carefully.
        is_planar = planar == 2
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating):
                mx = float(np.nanmax(rgb)) if rgb.size else 0.0
                if mx <= 1.0 + 1e-6:
                    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb, None, "L", ImageMeta()
    if arr.ndim == 3 and is_planar and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    elif (
        arr.ndim == 3
        and arr.shape[0] in (3, 4)
        and arr.shape[0] < arr.shape[1]
        and arr.shape[0] < arr.shape[2]
        and planar is None
    ):
        # Heuristic only when IFD lacks planarconfig and first axis is tiny.
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            mx = float(np.nanmax(arr)) if arr.size else 0.0
            if mx <= 1.0 + 1e-6:
                arr = (arr.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
            if info is not None and info.max > 0:
                arr = (
                    arr.astype(np.float32) / float(info.max) * 255.0
                ).clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[2] >= 4:
        return arr[:, :, :3].copy(), arr[:, :, 3].copy(), "RGBA", ImageMeta()
    if arr.shape[2] == 3:
        return arr.copy(), None, "RGB", ImageMeta()
    if arr.shape[2] == 1:
        rgb = np.repeat(arr, 3, axis=2)
        return rgb, None, "L", ImageMeta()
    raise ValueError(f"Unsupported TIFF channels {arr.shape[2]}")


def _load_via_psd(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None, str, ImageMeta]:
    try:
        from psd_tools import PSDImage
    except ImportError as exc:
        raise RuntimeError(
            "PSD/PSB input requires the psd-tools package. "
            'Install with: pip install "psd-tools"'
        ) from exc
    psd = PSDImage.open(path)
    composite = psd.composite()
    if composite is None:
        raise ValueError("PSD has no composite image")
    return _pil_to_rgba(composite)


def load_rgba_with_meta(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None, str, ImageMeta]:
    """Like ``load_rgba`` but also returns EXIF / ICC when present.

    Tries Pillow first, then format-specific readers (TIFF via tifffile,
    PSD/PSB via psd-tools) so web/CLI can accept formats browsers cannot decode.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    errors: list[str] = []

    if suffix in (".psd", ".psb"):
        try:
            return _load_via_psd(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"psd-tools: {exc}")
            # Fall through — Pillow can open some flat PSDs

    try:
        with Image.open(path) as img:
            img.load()
            return _pil_to_rgba(img.copy())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Pillow: {exc}")

    if suffix in (".tif", ".tiff"):
        try:
            return _load_via_tifffile(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"tifffile: {exc}")

    if suffix in (".psd", ".psb"):
        try:
            return _load_via_psd(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"psd-tools: {exc}")

    detail = "; ".join(errors) if errors else "unknown error"
    raise ValueError(f"Cannot load image {path.name}: {detail}")


def _normalize_fmt(fmt: str, long_edge: int) -> str:
    use = (fmt or "auto").lower().replace("-", "_")
    if use in ("jpg",):
        use = "jpeg"
    if use in ("tif",):
        use = "tiff"
    if use in ("jpeg2000", "j2k", "jpeg_2000"):
        use = "jp2"
    if use == "exif_jpeg":
        use = "jpeg_exif"
    if use == "auto":
        use = "tiff" if long_edge > LONG_EDGE_TIFF_THRESHOLD else "png"
    return use


def _ext_for_fmt(use: str) -> str:
    return {
        "png": ".png",
        "jpeg": ".jpg",
        "jpeg_exif": ".jpg",
        "webp": ".webp",
        "tiff": ".tif",
        "bmp": ".bmp",
        "tga": ".tga",
        "jp2": ".jp2",
        "psd": ".psd",
        "psb": ".psb",
    }.get(use, ".png")


def save_image(
    rgb: np.ndarray,
    path: Path,
    original_mode: str,
    alpha: np.ndarray | None,
    tiff_compression: str = "lzw",
    fmt: str = "auto",
    quality: int = 90,
    meta: ImageMeta | None = None,
) -> Path:
    """
    Save upscaled image. Auto uses TIFF when the long edge exceeds LONG_EDGE_TIFF_THRESHOLD.

    Formats: auto|png|jpeg|jpeg_exif|webp|tiff|bmp|tga|jp2|psd|psb
    TIFF compression: lzw|jpeg|none|deflate|zstd
    """
    path = Path(path)
    rgb_u8 = np.ascontiguousarray(rgb)
    if rgb_u8.ndim == 2:
        rgb_u8 = np.stack([rgb_u8, rgb_u8, rgb_u8], axis=-1)
    elif rgb_u8.shape[2] == 1:
        rgb_u8 = np.concatenate([rgb_u8, rgb_u8, rgb_u8], axis=2)
    rgb_u8 = np.ascontiguousarray(rgb_u8[:, :, :3], dtype=np.uint8)
    pil = Image.fromarray(rgb_u8, mode="RGB")

    if alpha is not None and original_mode in ("RGBA", "LA"):
        h, w = rgb_u8.shape[:2]
        a = np.asarray(alpha)
        if a.ndim == 3:
            a = a[:, :, 0]
        a = np.ascontiguousarray(a, dtype=np.uint8)
        alpha_pil = Image.fromarray(a, mode="L")
        if alpha_pil.size != (w, h):
            alpha_pil = alpha_pil.resize((w, h), Image.Resampling.LANCZOS)
        pil = pil.convert("RGBA")
        pil.putalpha(alpha_pil)

    long_edge = max(rgb.shape[:2])
    use = _normalize_fmt(fmt, long_edge)
    path = path.with_suffix(_ext_for_fmt(use))
    suffix = path.suffix.lower()
    meta = meta or ImageMeta()
    icc = meta.icc_profile

    if use == "tiff" or suffix in (".tif", ".tiff"):
        _save_tiff(pil, path, tiff_compression)
    elif use in ("jpeg", "jpeg_exif") or suffix in (".jpg", ".jpeg"):
        _save_jpeg(pil, path, quality=quality, preserve_exif=(use == "jpeg_exif"), meta=meta)
    elif use == "webp" or suffix == ".webp":
        kwargs: dict[str, Any] = {"format": "WEBP", "quality": int(quality)}
        if icc:
            kwargs["icc_profile"] = icc
        if pil.mode == "RGBA":
            kwargs["lossless"] = False
        pil.save(path, **kwargs)
    elif use == "bmp" or suffix == ".bmp":
        _save_bmp(pil, path)
    elif use == "tga" or suffix == ".tga":
        _save_tga(pil, path)
    elif use == "jp2" or suffix in (".jp2", ".j2k"):
        _save_jp2(pil, path, quality=quality)
    elif use in ("psd", "psb") or suffix in (".psd", ".psb"):
        _save_psd(pil, path, psb=(use == "psb" or suffix == ".psb"))
    else:
        kwargs = {"format": "PNG", "optimize": False}
        if icc:
            kwargs["icc_profile"] = icc
        pil.save(path, **kwargs)

    return path


def _save_bmp(pil: Image.Image, path: Path) -> None:
    """BMP via Pillow. Alpha is flattened onto black (BMP has no standard alpha)."""
    if pil.mode == "RGBA":
        bg = Image.new("RGB", pil.size, (0, 0, 0))
        bg.paste(pil, mask=pil.split()[-1])
        pil = bg
    elif pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil.save(path, format="BMP")


def _save_tga(pil: Image.Image, path: Path) -> None:
    """TGA via Pillow; keeps RGBA when present."""
    if pil.mode not in ("RGB", "RGBA"):
        pil = pil.convert("RGBA" if "A" in pil.getbands() else "RGB")
    pil.save(path, format="TGA")


def _save_tiff(pil: Image.Image, path: Path, tiff_compression: str) -> None:
    comp = (tiff_compression or "lzw").lower()
    if comp in ("deflate", "zip"):
        comp = "adobe_deflate"  # tifffile / PIL name variants
    try:
        import tifffile

        arr = np.array(pil)
        kwargs: dict[str, Any] = {"bigtiff": True}
        if comp == "none":
            pass
        elif comp in ("adobe_deflate", "deflate", "zip"):
            kwargs["compression"] = "adobe_deflate"
        elif comp == "zstd":
            kwargs["compression"] = "zstd"
        elif comp == "jpeg":
            kwargs["compression"] = "jpeg"
        elif comp == "lzw":
            kwargs["compression"] = "lzw"
        else:
            kwargs["compression"] = comp
        tifffile.imwrite(str(path), arr, **kwargs)
    except Exception:
        # Pillow fallback (limited compression choices)
        pil_comp = {
            "lzw": "tiff_lzw",
            "jpeg": "jpeg",
            "none": "raw",
            "adobe_deflate": "tiff_adobe_deflate",
            "deflate": "tiff_adobe_deflate",
            "zstd": "tiff_adobe_deflate",
        }.get(comp, "tiff_lzw")
        pil.save(path, format="TIFF", compression=pil_comp)


def _save_jpeg(
    pil: Image.Image,
    path: Path,
    *,
    quality: int,
    preserve_exif: bool,
    meta: ImageMeta,
) -> None:
    if pil.mode == "RGBA":
        pil = pil.convert("RGB")
    kwargs: dict[str, Any] = {
        "format": "JPEG",
        "quality": int(quality),
        "subsampling": 0,
    }
    if meta.icc_profile:
        kwargs["icc_profile"] = meta.icc_profile
    if preserve_exif and meta.exif:
        kwargs["exif"] = meta.exif
    pil.save(path, **kwargs)


def _save_jp2(pil: Image.Image, path: Path, *, quality: int) -> None:
    if pil.mode == "RGBA":
        # JP2 can do RGBA via Pillow when codec supports it; flatten if needed
        try:
            pil.save(
                path,
                format="JPEG2000",
                quality_mode="rates",
                quality_layers=[max(1, int(100 - quality) // 5 + 1)],
            )
            return
        except Exception:
            pil = pil.convert("RGB")
    try:
        pil.save(
            path,
            format="JPEG2000",
            quality_mode="rates",
            quality_layers=[max(1, int(100 - quality) // 5 + 1)],
        )
        return
    except Exception:
        pass
    # imagecodecs fallback
    try:
        import imagecodecs

        arr = np.array(pil.convert("RGB"), dtype=np.uint8)
        data = imagecodecs.jpeg2k_encode(arr, level=max(1, min(100, quality)))
        path.write_bytes(data)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "JPEG 2000 save failed — need Pillow with OpenJPEG or imagecodecs"
        ) from exc


def _save_psd(pil: Image.Image, path: Path, *, psb: bool) -> None:
    """Flattened single-layer PSD/PSB via the minimal writer."""
    rgb = pil.convert("RGB")
    _save_psd_minimal(np.array(rgb, dtype=np.uint8), path, psb=psb)


def _save_psd_minimal(rgb: np.ndarray, path: Path, *, psb: bool) -> None:
    """Write a minimal flattened 8-bit RGB PSD/PSB (no layers).

    PSD (version 1) is limited to 30000 px. PSB (version 2) allows up to 300000 px.
    """
    import struct

    h, w = rgb.shape[:2]
    edge = max(h, w)
    if psb:
        if edge > 300_000:
            raise ValueError("Image exceeds Photoshop PSB dimension limit (300000 px)")
    elif edge > 30_000:
        raise ValueError("PSD exceeds 30000 px; use PSB for larger images")
    version = 2 if psb else 1
    # File header (26 bytes): 8BPS, version, reserved(6), channels, height, width, depth, mode
    header = b"8BPS" + struct.pack(">H", version)
    header += b"\x00" * 6
    header += struct.pack(">H", 3)  # channels
    header += struct.pack(">II", h, w)
    header += struct.pack(">HH", 8, 3)  # 8-bit, RGB

    # Empty color mode / resources / layer sections (length-prefixed)
    def u32(n: int) -> bytes:
        return struct.pack(">I", n)

    def u64(n: int) -> bytes:
        return struct.pack(">Q", n)

    color_mode = u32(0)
    resources = u32(0)
    # Layer and mask information length (0 = flattened only)
    layers = u64(0) if psb else u32(0)

    # Image data: one compression field then planar R,G,B (raw)
    # Spec: compression is a single uint16 for the whole image data, then channel planes
    planes = b"".join(rgb[:, :, c].tobytes() for c in range(3))
    image_data = struct.pack(">H", 0) + planes  # 0 = raw

    path.write_bytes(header + color_mode + resources + layers + image_data)


def lanczos_resize(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Lanczos resize; prefers OpenCV when installed (via classical backend)."""
    try:
        from .algorithms.classical import resize_classical

        return resize_classical("lanczos3", img, target_w, target_h)
    except Exception:  # noqa: BLE001 — fall back to Pillow
        pil = Image.fromarray(img)
        pil = pil.resize((target_w, target_h), Image.LANCZOS)
        return np.array(pil)


DEFAULT_DISPLAY_MAX_EDGE = 4096
DEFAULT_THUMB_MAX_EDGE = 512


def write_display_preview(
    src: Path | str,
    dest: Path | str,
    *,
    max_edge: int = DEFAULT_DISPLAY_MAX_EDGE,
) -> Path:
    """Write a PNG preview with long edge ≤ ``max_edge`` (browser-safe).

    Full-resolution output stays on ``src`` / job result path for download.
    Uses ``load_rgba`` so TIFF/TGA/PSD/JP2 inputs preview correctly.
    """
    src_p, dest_p = Path(src), Path(dest)
    edge = max(16, int(max_edge))
    rgb, alpha, _mode = load_rgba(src_p)
    if alpha is not None:
        rgba = np.dstack([rgb, alpha])
        im = Image.fromarray(rgba, mode="RGBA")
    else:
        im = Image.fromarray(rgb, mode="RGB")
    if max(im.size) > edge:
        im = im.copy()
        im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest_p, format="PNG", optimize=False)
    return dest_p
