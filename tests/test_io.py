from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from upscaler.io_utils import (
    load_rgba,
    load_rgba_with_meta,
    save_image,
    write_display_preview,
)


def test_load_rgba_roundtrip(tmp_path: Path):
    rgba = np.zeros((16, 20, 4), dtype=np.uint8)
    rgba[..., 0] = 10
    rgba[..., 1] = 20
    rgba[..., 2] = 30
    rgba[..., 3] = 200
    src = tmp_path / "in.png"
    Image.fromarray(rgba, mode="RGBA").save(src)

    rgb, alpha, mode = load_rgba(src)
    assert mode == "RGBA"
    assert rgb.shape == (16, 20, 3)
    assert alpha is not None and alpha.shape == (16, 20)
    assert int(alpha[0, 0]) == 200

    # pre-sized alpha (already matches output)
    out_rgb = np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1)
    out_alpha = np.repeat(np.repeat(alpha, 2, axis=0), 2, axis=1)
    dst = tmp_path / "out.png"
    save_image(out_rgb, dst, "RGBA", out_alpha)
    loaded = Image.open(dst)
    assert loaded.mode == "RGBA"
    assert loaded.size == (40, 32)


def test_load_rgb_no_alpha(tmp_path: Path):
    rgb = np.full((8, 8, 3), 40, dtype=np.uint8)
    src = tmp_path / "rgb.png"
    Image.fromarray(rgb, mode="RGB").save(src)
    out, alpha, mode = load_rgba(src)
    assert alpha is None
    assert mode == "RGB"
    assert out.shape == (8, 8, 3)


def test_write_display_preview_caps_long_edge(tmp_path: Path):
    big = np.zeros((1200, 800, 3), dtype=np.uint8)
    big[..., 1] = 90
    src = tmp_path / "big.png"
    Image.fromarray(big, mode="RGB").save(src)
    dest = tmp_path / "disp.png"
    write_display_preview(src, dest, max_edge=512)
    with Image.open(dest) as im:
        assert max(im.size) <= 512
        assert im.size[0] == 341 or im.size[1] == 512  # 800×1200 → 341×512


def test_write_display_preview_noop_when_small(tmp_path: Path):
    src = tmp_path / "small.png"
    Image.fromarray(np.zeros((64, 48, 3), dtype=np.uint8), mode="RGB").save(src)
    dest = tmp_path / "disp.png"
    write_display_preview(src, dest, max_edge=4096)
    with Image.open(dest) as im:
        assert im.size == (48, 64)


def test_vector_inputs_rejected():
    from upscaler.io_utils import is_vector_input, require_raster_suffix

    assert is_vector_input("logo.svg")
    assert is_vector_input("logo.SVGZ")
    assert is_vector_input("x.png", "image/svg+xml")
    assert not is_vector_input("photo.png", "image/png")
    try:
        require_raster_suffix("icon.svg")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "Vector" in str(exc) or "vector" in str(exc).lower()


def test_load_tga_and_bmp(tmp_path: Path):
    rgb = np.full((12, 16, 3), 90, dtype=np.uint8)
    tga = tmp_path / "in.tga"
    bmp = tmp_path / "in.bmp"
    Image.fromarray(rgb, mode="RGB").save(tga)
    Image.fromarray(rgb, mode="RGB").save(bmp)
    for src in (tga, bmp):
        out, alpha, _mode = load_rgba(src)
        assert alpha is None
        assert out.shape == (12, 16, 3)
        prev = tmp_path / (src.stem + "_prev.png")
        write_display_preview(src, prev, max_edge=64)
        assert prev.is_file()


def _rgb(h: int = 32, w: int = 40) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = 10
    arr[..., 1] = 20
    arr[..., 2] = 30
    arr[4:12, 4:12] = 200
    return arr


def test_save_jpeg_and_webp(tmp_path: Path):
    rgb = _rgb()
    jpg = save_image(rgb, tmp_path / "a.png", "RGB", None, fmt="jpeg", quality=85)
    assert jpg.suffix.lower() in (".jpg", ".jpeg")
    assert jpg.is_file()
    webp = save_image(rgb, tmp_path / "b.png", "RGB", None, fmt="webp", quality=80)
    assert webp.suffix == ".webp"
    assert webp.is_file()


def test_save_tiff_deflate(tmp_path: Path):
    rgb = _rgb()
    path = save_image(
        rgb, tmp_path / "t.png", "RGB", None, fmt="tiff", tiff_compression="deflate"
    )
    assert path.suffix.lower() in (".tif", ".tiff")
    assert path.is_file()
    with Image.open(path) as im:
        assert im.size == (40, 32)


@pytest.mark.parametrize("comp", ["lzw", "none", "zstd"])
def test_save_tiff_variants(tmp_path: Path, comp: str):
    rgb = _rgb()
    try:
        path = save_image(
            rgb, tmp_path / f"t_{comp}.png", "RGB", None, fmt="tiff", tiff_compression=comp
        )
    except Exception as exc:  # noqa: BLE001
        if comp == "zstd":
            pytest.skip(f"zstd TIFF unavailable: {exc}")
        raise
    assert path.is_file()


def test_save_jpeg_exif_preserves_exif(tmp_path: Path):
    src = tmp_path / "src.jpg"
    Image.fromarray(_rgb(), mode="RGB").save(src, format="JPEG", quality=90, exif=b"Exif\x00\x00" + b"\x00" * 8)
    # Build a slightly richer EXIF via Pillow getexif if possible
    im = Image.open(src)
    exif = im.getexif()
    exif[271] = "UpscaleTest"  # Make
    im.save(src, format="JPEG", quality=90, exif=exif.tobytes())

    _rgb_arr, _a, _m, meta = load_rgba_with_meta(src)
    assert meta.exif
    out = save_image(
        _rgb(),
        tmp_path / "out.jpg",
        "RGB",
        None,
        fmt="jpeg_exif",
        quality=90,
        meta=meta,
    )
    with Image.open(out) as loaded:
        got = loaded.getexif()
        assert got.get(271) == "UpscaleTest"


def test_save_jp2(tmp_path: Path):
    rgb = _rgb()
    try:
        path = save_image(rgb, tmp_path / "a.png", "RGB", None, fmt="jp2", quality=80)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert path.suffix == ".jp2"
    assert path.is_file() and path.stat().st_size > 32


@pytest.mark.parametrize("fmt", ["bmp", "tga"])
def test_save_bmp_and_tga(tmp_path: Path, fmt: str):
    rgb = _rgb()
    path = save_image(rgb, tmp_path / "a.png", "RGB", None, fmt=fmt)
    assert path.suffix == f".{fmt}"
    assert path.is_file() and path.stat().st_size > 32
    with Image.open(path) as im:
        assert im.size == (40, 32)


def test_save_tga_keeps_alpha(tmp_path: Path):
    rgb = _rgb(8, 8)
    alpha = np.full((8, 8), 128, dtype=np.uint8)
    path = save_image(rgb, tmp_path / "a.png", "RGBA", alpha, fmt="tga")
    assert path.suffix == ".tga"
    with Image.open(path) as im:
        assert im.mode in ("RGBA", "RGB")
        if im.mode == "RGBA":
            assert min(im.getchannel("A").tobytes()) < 255


@pytest.mark.parametrize("fmt", ["psd", "psb"])
def test_save_psd_psb_flat(tmp_path: Path, fmt: str):
    rgb = _rgb(16, 16)
    path = save_image(rgb, tmp_path / "a.png", "RGB", None, fmt=fmt)
    assert path.suffix == f".{fmt}"
    assert path.is_file()
    data = path.read_bytes()
    assert data[:4] == b"8BPS"
    # Version 1 = PSD, 2 = PSB
    ver = int.from_bytes(data[4:6], "big")
    assert ver == (2 if fmt == "psb" else 1)


def test_psd_size_limit_psb_allows_larger(tmp_path: Path):
    rgb = np.zeros((1, 30001, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="30000"):
        save_image(rgb, tmp_path / "big.png", "RGB", None, fmt="psd")
    path = save_image(rgb, tmp_path / "big.png", "RGB", None, fmt="psb")
    assert path.suffix == ".psb"
    assert path.read_bytes()[:4] == b"8BPS"
