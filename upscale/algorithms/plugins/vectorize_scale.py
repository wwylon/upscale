"""Vectorize (VTracer or PyAutoTrace), then rasterize at the target size.

Best for logos / line art / flat graphics — not photographs.
Optional dependency: ``pip install 'upscale[vectorize]'``
(vtracer + pyautotrace + resvg_py).

SVG is produced internally; users still cannot upload SVG as input.
"""

from __future__ import annotations

import io
import logging
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

from upscale.algorithms import register_algorithm
from upscale.algorithms.classical import resize_classical

logger = logging.getLogger(__name__)

_DEFAULT_TRACE_EDGE = 1024


def _to_rgba(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        rgb = np.stack([img, img, img], axis=-1)
        a = np.full(img.shape, 255, dtype=np.uint8)
        return np.dstack([rgb, a])
    if img.shape[2] == 3:
        a = np.full(img.shape[:2], 255, dtype=np.uint8)
        return np.dstack([img, a])
    return img[:, :, :4]


def _maybe_downscale(rgb: np.ndarray, max_trace_edge: int) -> np.ndarray:
    """Downscale for tracing speed on large images."""
    cap = max(256, int(max_trace_edge))
    h, w = rgb.shape[:2]
    edge = max(h, w)
    if edge <= cap:
        return rgb
    scale = cap / edge
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))
    return resize_classical("bilinear", rgb, tw, th)


def _trace_svg_vtracer(
    rgb: np.ndarray,
    *,
    color_precision: int = 6,
    filter_speckle: int = 4,
    corner_threshold: int = 60,
    mode: str = "spline",
) -> str:
    import vtracer

    rgba = _to_rgba(rgb)
    h, w = rgba.shape[:2]
    pixels = [tuple(int(x) for x in px) for px in rgba.reshape(-1, 4)]
    path_mode = mode if mode in ("spline", "polygon") else "spline"
    return vtracer.convert_pixels_to_svg(
        pixels,
        (w, h),
        colormode="color",
        hierarchical="stacked",
        mode=path_mode,
        filter_speckle=int(filter_speckle),
        color_precision=int(color_precision),
        layer_difference=16,
        corner_threshold=int(corner_threshold),
        length_threshold=4.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=3,
    )


def _trace_svg_autotrace(
    rgb: np.ndarray,
    *,
    color_precision: int = 6,
    filter_speckle: int = 4,
    corner_threshold: int = 60,
) -> str:
    """Trace with PyAutoTrace (classic AutoTrace). Returns SVG text."""
    try:
        from autotrace import Bitmap, VectorFormat
    except ImportError as exc:
        raise ImportError(
            "autotrace engine requires pyautotrace. "
            "Install with: pip install 'upscale[vectorize]'"
        ) from exc

    # Map shared UI knobs onto AutoTrace TraceOptions.
    # color_precision 1..8 → color_count 2..256 (0 = AutoTrace default/auto)
    color_count = int(2 ** max(1, min(8, color_precision)))
    despeckle_level = int(np.clip(filter_speckle, 0, 20))
    # Our UI uses 0–180 with default 60; AutoTrace default is 100.
    at_corner = float(np.clip(corner_threshold, 0, 180))

    bitmap = Bitmap(np.ascontiguousarray(rgb, dtype=np.uint8))
    vector = bitmap.trace(
        color_count=color_count,
        corner_threshold=at_corner,
        despeckle_level=despeckle_level,
        filter_iterations=4,
    )
    raw = vector.encode(VectorFormat.SVG)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _trace_svg(
    rgb: np.ndarray,
    *,
    engine: str = "vtracer",
    color_precision: int = 6,
    filter_speckle: int = 4,
    corner_threshold: int = 60,
    mode: str = "spline",
) -> str:
    eng = (engine or "vtracer").strip().lower()
    if eng == "autotrace":
        return _trace_svg_autotrace(
            rgb,
            color_precision=color_precision,
            filter_speckle=filter_speckle,
            corner_threshold=corner_threshold,
        )
    try:
        import vtracer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "vtracer engine requires vtracer. "
            "Install with: pip install 'upscale[vectorize]'"
        ) from exc
    return _trace_svg_vtracer(
        rgb,
        color_precision=color_precision,
        filter_speckle=filter_speckle,
        corner_threshold=corner_threshold,
        mode=mode,
    )


def _fit_rgb(arr: np.ndarray, tw: int, th: int) -> np.ndarray:
    rgb = np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
    if rgb.shape[0] != th or rgb.shape[1] != tw:
        rgb = resize_classical("lanczos3", rgb, tw, th)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _rasterize_svg(svg: str, tw: int, th: int) -> np.ndarray:
    """Rasterize SVG to RGB uint8 at tw×th.

    Prefers ``resvg_py`` (works on Windows without Cairo). Falls back to
    cairosvg / reportlab / a limited Pillow path.
    """
    data = svg.encode("utf-8")

    # 1) resvg (Rust) — primary, Windows-friendly
    try:
        import resvg_py

        png = resvg_py.svg_to_bytes(svg_string=svg, width=tw, height=th)
        if png:
            with Image.open(io.BytesIO(png)) as im:
                return _fit_rgb(np.asarray(im.convert("RGB")), tw, th)
    except Exception as exc:  # noqa: BLE001
        logger.debug("resvg_py rasterize failed: %s", exc)

    # 2) cairosvg
    try:
        import cairosvg

        png = cairosvg.svg2png(bytestring=data, output_width=tw, output_height=th)
        with Image.open(io.BytesIO(png)) as im:
            return _fit_rgb(np.asarray(im.convert("RGB")), tw, th)
    except Exception as exc:  # noqa: BLE001
        logger.debug("cairosvg rasterize failed: %s", exc)

    # 3) svglib + reportlab (needs rlPyCairo / _rl_renderPM)
    try:
        from reportlab.graphics import renderPM
        from svglib.svglib import svg2rlg

        drawing = svg2rlg(io.BytesIO(data))
        if drawing is not None:
            sx = tw / max(float(drawing.width or 1), 1)
            sy = th / max(float(drawing.height or 1), 1)
            drawing.width, drawing.height = tw, th
            drawing.scale(sx, sy)
            pil = renderPM.drawToPIL(drawing)
            return _fit_rgb(np.asarray(pil.convert("RGB")), tw, th)
    except Exception as exc:  # noqa: BLE001
        logger.debug("reportlab rasterize failed: %s", exc)

    # 4) Limited Pillow path (M/L/H/V + translate) — last resort
    logger.warning(
        "SVG rasterizer falling back to limited Pillow path; "
        "install resvg_py for correct Vectorize & scale output "
        "(pip install 'upscale[vectorize]')."
    )
    return _rasterize_svg_pillow(svg, tw, th)


def _parse_color(style: str | None, fill: str | None) -> tuple[int, int, int] | None:
    raw = (fill or "").strip()
    if style:
        m = re.search(r"fill\s*:\s*([^;]+)", style)
        if m:
            raw = m.group(1).strip()
    if not raw or raw in ("none", "transparent"):
        return None
    if raw.startswith("#") and len(raw) in (4, 7):
        if len(raw) == 4:
            return tuple(int(c * 2, 16) for c in raw[1:])  # type: ignore[return-value]
        return (
            int(raw[1:3], 16),
            int(raw[3:5], 16),
            int(raw[5:7], 16),
        )
    if raw.startswith("rgb"):
        nums = re.findall(r"\d+", raw)
        if len(nums) >= 3:
            return int(nums[0]), int(nums[1]), int(nums[2])
    return (0, 0, 0)


def _parse_translate(transform: str | None) -> tuple[float, float]:
    if not transform:
        return 0.0, 0.0
    m = re.search(
        r"translate\(\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*[, ]\s*"
        r"([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\)",
        transform,
    )
    if not m:
        return 0.0, 0.0
    return float(m.group(1)), float(m.group(2))


def _path_to_points(
    d: str, sx: float, sy: float, ox: float = 0.0, oy: float = 0.0
) -> list[tuple[float, float]]:
    """Subset of SVG path commands including cubic Bezier (C/c) sampling."""
    tokens = re.findall(r"[MmLlHhVvCcZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d)
    pts: list[tuple[float, float]] = []
    cx = cy = 0.0
    i = 0
    cmd = "M"

    def emit(x: float, y: float) -> None:
        pts.append(((x + ox) * sx, (y + oy) * sy))

    while i < len(tokens):
        t = tokens[i]
        if re.match(r"[A-Za-z]", t):
            cmd = t
            i += 1
            continue
        if cmd in ("M", "L", "m", "l"):
            if i + 1 >= len(tokens):
                break
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if cmd.islower():
                x, y = cx + x, cy + y
            cx, cy = x, y
            emit(cx, cy)
            if cmd in ("M", "m"):
                cmd = "l" if cmd.islower() else "L"
        elif cmd in ("H", "h"):
            x = float(tokens[i])
            i += 1
            if cmd == "h":
                x = cx + x
            cx = x
            emit(cx, cy)
        elif cmd in ("V", "v"):
            y = float(tokens[i])
            i += 1
            if cmd == "v":
                y = cy + y
            cy = y
            emit(cx, cy)
        elif cmd in ("C", "c"):
            if i + 5 >= len(tokens):
                break
            x1, y1 = float(tokens[i]), float(tokens[i + 1])
            x2, y2 = float(tokens[i + 2]), float(tokens[i + 3])
            x, y = float(tokens[i + 4]), float(tokens[i + 5])
            i += 6
            if cmd == "c":
                x1, y1 = cx + x1, cy + y1
                x2, y2 = cx + x2, cy + y2
                x, y = cx + x, cy + y
            # Sample cubic Bezier
            for s in range(1, 9):
                t_ = s / 8.0
                u = 1.0 - t_
                bx = (
                    u * u * u * cx
                    + 3 * u * u * t_ * x1
                    + 3 * u * t_ * t_ * x2
                    + t_ * t_ * t_ * x
                )
                by = (
                    u * u * u * cy
                    + 3 * u * u * t_ * y1
                    + 3 * u * t_ * t_ * y2
                    + t_ * t_ * t_ * y
                )
                emit(bx, by)
            cx, cy = x, y
        elif cmd in ("Z", "z"):
            if pts:
                pts.append(pts[0])
            i += 1
        else:
            i += 1
    return pts


def _rasterize_svg_pillow(svg: str, tw: int, th: int) -> np.ndarray:
    root = ET.fromstring(svg)
    vb = root.get("viewBox")
    if vb:
        parts = [float(x) for x in vb.replace(",", " ").split()]
        vw, vh = parts[2], parts[3]
    else:
        vw = float(root.get("width", tw) or tw)
        vh = float(root.get("height", th) or th)
    sx, sy = tw / max(vw, 1e-6), th / max(vh, 1e-6)
    canvas = Image.new("RGB", (tw, th), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for el in root.iter():
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        color = _parse_color(el.get("style"), el.get("fill"))
        if color is None:
            continue
        ox, oy = _parse_translate(el.get("transform"))
        if tag == "path" and el.get("d"):
            pts = _path_to_points(el.get("d") or "", sx, sy, ox, oy)
            if len(pts) >= 3:
                draw.polygon(pts, fill=color)
        elif tag == "polygon" and el.get("points"):
            nums = [
                float(x)
                for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", el.get("points") or "")
            ]
            pts = [
                ((nums[i] + ox) * sx, (nums[i + 1] + oy) * sy)
                for i in range(0, len(nums) - 1, 2)
            ]
            if len(pts) >= 3:
                draw.polygon(pts, fill=color)
        elif tag == "rect":
            x = (float(el.get("x", 0)) + ox) * sx
            y = (float(el.get("y", 0)) + oy) * sy
            w = float(el.get("width", 0)) * sx
            h = float(el.get("height", 0)) * sy
            draw.rectangle([x, y, x + w, y + h], fill=color)
    return np.asarray(canvas, dtype=np.uint8)


def vectorize_scale_resize(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    engine: str = "vtracer",
    color_precision: int = 6,
    filter_speckle: int = 4,
    corner_threshold: int = 60,
    mode: str = "spline",
    max_trace_edge: int = _DEFAULT_TRACE_EDGE,
) -> np.ndarray:
    """Trace to SVG then rasterize at target size.

    Gray / alpha (HxW) inputs return HxW; RGB returns HxWx3 uint8.
    ``engine`` is ``vtracer`` (default) or ``autotrace`` (PyAutoTrace).
    """
    eng = (engine or "vtracer").strip().lower()
    if eng == "autotrace":
        try:
            import autotrace  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "vectorize_scale engine=autotrace requires pyautotrace. "
                "Install with: pip install 'upscale[vectorize]'"
            ) from exc
    else:
        try:
            import vtracer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "vectorize_scale engine=vtracer requires vtracer. "
                "Install with: pip install 'upscale[vectorize]'"
            ) from exc

    raw = np.asarray(img, dtype=np.uint8)
    gray_in = raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] == 1)
    if gray_in:
        g = raw[:, :, 0] if raw.ndim == 3 else raw
        rgb = np.stack([g, g, g], axis=-1)
    else:
        rgb = raw[:, :, :3]
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    tw, th = int(target_w), int(target_h)

    small = _maybe_downscale(rgb, max_trace_edge)
    svg = _trace_svg(
        small,
        engine=eng,
        color_precision=color_precision,
        filter_speckle=filter_speckle,
        corner_threshold=corner_threshold,
        mode=mode,
    )
    out = _rasterize_svg(svg, tw, th)
    out = np.ascontiguousarray(out[:, :, :3], dtype=np.uint8)
    if gray_in:
        return out[:, :, 0]
    return out


def _entry(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    return vectorize_scale_resize(img, tw, th)


register_algorithm(
    "vectorize_scale",
    _entry,
    overwrite=True,
    label="Vectorize & scale",
    group="paper",
    description=(
        "Trace to SVG (VTracer or PyAutoTrace), then rasterize at the target size. "
        "Best for logos and flat graphics — not photos. "
        "Requires: pip install 'upscale[vectorize]'."
    ),
)
