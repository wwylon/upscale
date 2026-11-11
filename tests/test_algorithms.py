"""Classical algorithms + plugin registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from upscale.algorithms import (
    list_algorithm_meta,
    list_algorithms,
    register_algorithm,
    resize_with_algorithm,
    unregister_algorithm,
)
from upscale.algorithms.classical import BUILTIN_ALGORITHMS, backend_name
from upscale.algorithms.plugins.texture_residual import texture_residual_resize
from upscale.config import UpscaleConfig
from upscale.engine import execute


def _checker(h: int = 16, w: int = 16) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[0 : h // 2, 0 : w // 2] = (255, 0, 0)
    img[h // 2 :, w // 2 :] = (0, 255, 0)
    return img


@pytest.mark.parametrize("name", list(BUILTIN_ALGORITHMS))
def test_classical_output_size(name: str):
    img = _checker()
    out = resize_with_algorithm(name, img, 64, 48)
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8


def test_nearest_preserves_block_colors():
    img = _checker(8, 8)
    out = resize_with_algorithm("nearest", img, 16, 16)
    # Top-left quadrant stays red-ish
    assert int(out[2, 2, 0]) == 255
    assert int(out[2, 2, 1]) == 0


def test_list_includes_builtins():
    names = list_algorithms()
    for n in BUILTIN_ALGORITHMS:
        assert n in names
    assert backend_name() in ("opencv", "pillow")


def test_plugins_auto_discovered():
    names = list_algorithms()
    assert "texture_residual" in names
    assert "vectorize_scale" in names
    meta = {m.name: m for m in list_algorithm_meta()}
    assert meta["texture_residual"].group == "paper"
    assert meta["vectorize_scale"].group == "paper"


def test_unknown_algorithm_fails_cleanly():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        resize_with_algorithm("not_a_real_algo_xyz", _checker(), 32, 32)


def test_plugin_registry_hook():
    def double_then_nearest(img: np.ndarray, tw: int, th: int) -> np.ndarray:
        # Identity-ish plugin: nearest via builtin
        return resize_with_algorithm("nearest", img, tw, th)

    register_algorithm("test_plugin_double", double_then_nearest, overwrite=True)
    try:
        assert "test_plugin_double" in list_algorithms()
        out = resize_with_algorithm("test_plugin_double", _checker(), 32, 32)
        assert out.shape == (32, 32, 3)
    finally:
        unregister_algorithm("test_plugin_double")


def test_texture_residual_shape_and_strength_zero():
    img = _checker(24, 24)
    out = resize_with_algorithm("texture_residual", img, 48, 48)
    assert out.shape == (48, 48, 3)
    assert out.dtype == np.uint8
    # strength 0 ≈ Lanczos structure only
    zero = texture_residual_resize(img, 48, 48, strength=0.0)
    lanczos = resize_with_algorithm("lanczos3", img, 48, 48)
    assert np.mean(np.abs(zero.astype(np.int16) - lanczos.astype(np.int16))) < 1.0


def test_texture_residual_no_torch_import():
    from upscale.algorithms.plugins import texture_residual as tr

    src = Path(tr.__file__).read_text(encoding="utf-8")
    assert "import torch" not in src
    assert "from torch" not in src
    # Runtime path must not need torch either
    out = tr.texture_residual_resize(_checker(8, 8), 16, 16)
    assert out.shape == (16, 16, 3)


def test_vectorize_scale_synthetic_icon():
    vtracer = pytest.importorskip("vtracer")
    _ = vtracer
    # Tiny flat icon (hard edges, few colors)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :] = (240, 240, 240)
    img[8:24, 8:24] = (20, 80, 200)
    img[12:20, 12:20] = (255, 200, 40)
    out = resize_with_algorithm("vectorize_scale", img, 64, 64)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_execute_classical_end_to_end(tmp_path: Path):
    src = tmp_path / "in.png"
    Image.fromarray(_checker(32, 40), mode="RGB").save(src)
    out = tmp_path / "out.png"
    cfg = UpscaleConfig(
        input=src,
        output=out,
        scale=2.0,
        algorithm="bilinear",
        quiet=True,
        format="png",
    )
    path = execute(cfg, echo=lambda _m: None)
    assert path.is_file()
    with Image.open(path) as im:
        assert im.size == (80, 64)


def test_execute_texture_residual(tmp_path: Path):
    src = tmp_path / "in.png"
    Image.fromarray(_checker(32, 32), mode="RGB").save(src)
    out = tmp_path / "out.png"
    cfg = UpscaleConfig(
        input=src,
        output=out,
        scale=2.0,
        algorithm="texture_residual",
        quiet=True,
        format="png",
    )
    path = execute(cfg, echo=lambda _m: None)
    assert path.is_file()
    with Image.open(path) as im:
        assert im.size == (64, 64)


def test_execute_texture_residual_rgba(tmp_path: Path):
    src = tmp_path / "in.png"
    rgba = np.dstack([_checker(32, 40), np.full((32, 40), 200, dtype=np.uint8)])
    Image.fromarray(rgba, mode="RGBA").save(src)
    out = tmp_path / "out.png"
    cfg = UpscaleConfig(
        input=src,
        output=out,
        scale=2.0,
        algorithm="texture_residual",
        quiet=True,
        format="png",
    )
    path = execute(cfg, echo=lambda _m: None)
    assert path.is_file()
    with Image.open(path) as im:
        assert im.mode == "RGBA"
        assert im.size == (80, 64)


def test_execute_vectorize_scale_rgba(tmp_path: Path):
    pytest.importorskip("vtracer")
    src = tmp_path / "in.png"
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :] = (240, 240, 240)
    rgb[8:24, 8:24] = (20, 80, 200)
    rgba = np.dstack([rgb, np.full((32, 32), 220, dtype=np.uint8)])
    Image.fromarray(rgba, mode="RGBA").save(src)
    out = tmp_path / "out.png"
    path = execute(
        UpscaleConfig(
            input=src,
            output=out,
            scale=2.0,
            algorithm="vectorize_scale",
            quiet=True,
            format="png",
        ),
        echo=lambda _m: None,
    )
    assert path.is_file()
    with Image.open(path) as im:
        assert im.mode == "RGBA"
        assert im.size == (64, 64)


def test_vectorize_params_change_output():
    pytest.importorskip("vtracer")
    from upscale.algorithms.plugins.vectorize_scale import vectorize_scale_resize
    from upscale.schema import VectorizeSettings

    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[:, :] = (245, 245, 245)
    img[8:40, 8:40] = (40, 120, 200)
    img[16:32, 16:32] = (220, 60, 40)
    flat = vectorize_scale_resize(
        img, 96, 96, color_precision=2, filter_speckle=8, mode="polygon"
    )
    rich = vectorize_scale_resize(
        img, 96, 96, color_precision=8, filter_speckle=0, mode="spline"
    )
    assert flat.shape == rich.shape == (96, 96, 3)
    # Settings round-trip on JobSettings
    vz = VectorizeSettings(
        engine="autotrace", color_precision=3, mode="polygon", max_trace_edge=512
    )
    assert vz.engine == "autotrace"
    assert vz.color_precision == 3
    assert vz.mode == "polygon"


def test_vectorize_autotrace_engine():
    pytest.importorskip("autotrace")
    from upscale.algorithms.plugins.vectorize_scale import vectorize_scale_resize

    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:, :] = (250, 250, 250)
    img[8:32, 8:32] = (20, 90, 200)
    out = vectorize_scale_resize(
        img, 80, 80, engine="autotrace", color_precision=4, filter_speckle=2
    )
    assert out.shape == (80, 80, 3)
    assert out.dtype == np.uint8
