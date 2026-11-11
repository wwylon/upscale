"""Strip flush/carry must match a naive full-frame float accumulator."""

from __future__ import annotations

import numpy as np

from upscale.tiling import tiled_upscale


def _full_canvas_upscale(img, infer_fn, model_scale, tile, overlap):
    """Reference: accumulate entire float canvas then normalize."""
    from upscale.tiling import _tile_starts, make_tile_weight

    H, W = img.shape[:2]
    C = img.shape[2] if img.ndim == 3 else 1
    out_H, out_W = H * model_scale, W * model_scale
    canvas = np.zeros((out_H, out_W, C), dtype=np.float32)
    weight = np.zeros((out_H, out_W), dtype=np.float32)
    ys = _tile_starts(H, tile, overlap)
    xs = _tile_starts(W, tile, overlap)
    for y in ys:
        for x in xs:
            y2, x2 = min(H, y + tile), min(W, x + tile)
            patch = img[y:y2, x:x2]
            up = infer_fn(patch)
            oh, ow = up.shape[:2]
            oy, ox = y * model_scale, x * model_scale
            ft = y > 0
            fb = y2 < H
            fl = x > 0
            fr = x2 < W
            mask = make_tile_weight(
                oh, ow, overlap * model_scale,
                feather_top=ft, feather_bottom=fb, feather_left=fl, feather_right=fr,
            )
            canvas[oy : oy + oh, ox : ox + ow] += up.astype(np.float32) * mask[..., None]
            weight[oy : oy + oh, ox : ox + ow] += mask
    w = np.maximum(weight, 1e-8)[..., None]
    return np.clip(canvas / w, 0, 255).astype(np.uint8)


def test_strip_matches_full_canvas_patterned():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (96, 80, 3), dtype=np.uint8)

    def infer(patch):
        # 2× nearest
        return np.repeat(np.repeat(patch, 2, axis=0), 2, axis=1)

    strip = tiled_upscale(
        img, infer, model_scale=2, tile=32, overlap=8, show_progress=False
    )
    full = _full_canvas_upscale(img, infer, model_scale=2, tile=32, overlap=8)
    assert strip.shape == full.shape
    # Allow tiny numeric differences from strip band order
    assert np.mean(np.abs(strip.astype(np.int16) - full.astype(np.int16))) < 1.0
