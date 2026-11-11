from __future__ import annotations

import numpy as np

from upscale.filters import (
    any_post_filters,
    any_pre_filters,
    apply_post_filters,
    apply_pre_filters,
    estimate_time_factor,
)
from upscale.schema import FilterSettings


def test_filters_change_pixels():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = 200
    settings = FilterSettings(pre_denoise=0.5, jpeg_clean=0.4, edge=0.6, grain=0.3)
    pre = apply_pre_filters(img, settings)
    post = apply_post_filters(pre, settings)
    assert pre.shape == img.shape
    assert post.shape == img.shape
    assert estimate_time_factor(settings) > 1.0


def test_filters_noop():
    img = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    out = apply_pre_filters(img, FilterSettings())
    assert np.array_equal(out, img)


def test_new_filters_change_pixels_and_time():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
    # Soften so deblur has something to do
    soft = apply_pre_filters(img, FilterSettings(pre_denoise=0.3))
    deblurred = apply_pre_filters(soft, FilterSettings(deblur=0.7))
    assert deblurred.shape == img.shape
    assert not np.array_equal(deblurred, soft)

    post = apply_post_filters(
        img,
        FilterSettings(color_contrast=0.6, halo_reduce=0.5),
    )
    assert post.shape == img.shape
    assert not np.array_equal(post, img)

    s = FilterSettings(deblur=0.5, color_contrast=0.4, halo_reduce=0.3)
    assert estimate_time_factor(s) > 1.0
    assert any_pre_filters(FilterSettings(deblur=0.2))
    assert any_post_filters(FilterSettings(halo_reduce=0.2))
    assert not any_pre_filters(FilterSettings())
    assert not any_post_filters(FilterSettings())


def test_face_restore_ignored_until_wired():
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    s = FilterSettings(face_restore=0.8)
    # Not applied yet — output identical; ETA also ignores stub
    assert np.array_equal(apply_post_filters(img, s), img)
    assert estimate_time_factor(s) == 1.0
