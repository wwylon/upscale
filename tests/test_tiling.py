import numpy as np

from upscale.tiling import _tile_starts, make_tile_weight, tiled_upscale


def test_tile_starts_cover_full_length():
    length, tile, overlap = 200, 64, 16
    starts = _tile_starts(length, tile, overlap)
    assert starts[0] == 0
    # every pixel covered
    covered = np.zeros(length, dtype=bool)
    for s in starts:
        covered[s : min(s + tile, length)] = True
    assert covered.all()
    # no duplicate consecutive starts
    assert len(starts) == len(set(starts))


def test_tile_starts_small_image():
    starts = _tile_starts(40, 64, 16)
    assert starts == [0]


def test_border_feather_disabled_keeps_weight_one():
    mask = make_tile_weight(
        32, 32, 8,
        feather_top=False,
        feather_bottom=False,
        feather_left=False,
        feather_right=False,
    )
    assert float(mask.min()) == 1.0


def test_interior_feather_ramps_to_zero():
    mask = make_tile_weight(32, 32, 8)
    assert float(mask[0, 0]) == 0.0
    assert float(mask[16, 16]) == 1.0


def _fake_infer_scale4(patch: np.ndarray) -> np.ndarray:
    """Nearest-neighbor 4x stand-in for a real SR model."""
    return np.repeat(np.repeat(patch, 4, axis=0), 4, axis=1)


def test_tiled_upscale_no_black_borders():
    gray = np.full((96, 96, 3), 128, dtype=np.uint8)
    out = tiled_upscale(
        gray,
        _fake_infer_scale4,
        model_scale=4,
        tile=32,
        overlap=8,
        show_progress=False,
    )
    assert out.shape == (384, 384, 3)
    border = np.concatenate(
        [
            out[0, :, :].ravel(),
            out[-1, :, :].ravel(),
            out[:, 0, :].ravel(),
            out[:, -1, :].ravel(),
        ]
    )
    interior = out[40:-40, 40:-40]
    assert abs(float(border.mean()) - float(interior.mean())) < 1.5
    assert float(border.min()) > 100  # not black
