"""Minimal PNGs that current Pillow versions accept."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


def tiny_png_bytes(size: int = 1, color: tuple[int, int, int] = (120, 120, 120)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


TINY_PNG = tiny_png_bytes(1)
