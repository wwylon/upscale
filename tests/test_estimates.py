from __future__ import annotations

from upscale.estimates import (
    estimate_file_size_bytes,
    estimate_time_seconds,
    megapixels,
    model_fits_vram,
    output_dimensions,
    print_sizes,
    resource_check,
)
from upscale.help_content import get_help, list_topics
from upscale.schema import ResourceStatus


def test_output_dimensions_scale():
    w, h, s = output_dimensions(100, 50, scale=4)
    assert (w, h, s) == (400, 200, 4.0)


def test_output_dimensions_long_edge():
    w, h, s = output_dimensions(100, 50, target_size=1000)
    assert w == 1000 and h == 500 and s == 10.0


def test_print_sizes():
    sizes = print_sizes(3000, 2000)
    assert len(sizes) == 4
    assert sizes[0].dpi == 72


def test_auto_file_size_uses_tiff_threshold():
    from upscale.io_utils import LONG_EDGE_TIFF_THRESHOLD

    edge = LONG_EDGE_TIFF_THRESHOLD + 1
    assert estimate_file_size_bytes(edge, 10, "auto") == estimate_file_size_bytes(edge, 10, "tiff")
    assert estimate_file_size_bytes(LONG_EDGE_TIFF_THRESHOLD, 10, "auto") == estimate_file_size_bytes(
        LONG_EDGE_TIFF_THRESHOLD, 10, "png"
    )


def test_file_size_and_time():
    lo, hi = estimate_file_size_bytes(1000, 1000, "png")
    assert 0 < lo <= hi
    tlo, thi = estimate_time_seconds(256, 256, 4.0, device="cpu")
    assert 0 < tlo <= thi


def test_time_includes_load_and_download_overhead():
    ready_lo, ready_hi = estimate_time_seconds(
        512, 512, 4.0, device="cpu", model_ready=True
    )
    cold_lo, cold_hi = estimate_time_seconds(
        512,
        512,
        4.0,
        device="cpu",
        model_ready=False,
        model_bytes=67_000_000,
    )
    assert cold_lo > ready_lo
    assert cold_hi > ready_hi


def test_time_filter_factor_increases():
    base_lo, _ = estimate_time_seconds(256, 256, 4.0, device="cpu", filter_factor=1.0)
    filt_lo, _ = estimate_time_seconds(256, 256, 4.0, device="cpu", filter_factor=1.5)
    assert filt_lo > base_lo


def test_megapixels():
    assert megapixels(1000, 1000) == 1.0


def test_resource_check_blocks_low_ram():
    rc = resource_check(
        out_w=20000,
        out_h=20000,
        tile=512,
        fp16=True,
        model_bytes=67_000_000,
        vram_free_mb=8000,
        ram_available_mb=100,
        disk_free_mb=100_000,
        model_ready=True,
    )
    assert rc.blocked
    assert any(r.status == ResourceStatus.not_enough for r in rc.rows)


def test_model_fits_vram_levels():
    assert model_fits_vram(None, None) == "ok"
    assert model_fits_vram(67_000_000, 12000, tile=512, fp16=True) == "ok"
    # ~160 MB weights → ~240 MB need; 50 MB free → spill
    assert model_fits_vram(160_000_000, 50, tile=512, fp16=False) == "spill"
    # ~100 MB need with 90 free → tight
    assert model_fits_vram(67_000_000, 90, tile=512, fp16=False) == "tight"


def test_resource_check_vram_spill_blocks():
    rc = resource_check(
        out_w=1024,
        out_h=1024,
        tile=1024,
        fp16=False,
        model_bytes=200_000_000,
        vram_free_mb=40,
        ram_available_mb=16000,
        disk_free_mb=100_000,
        model_ready=True,
    )
    assert rc.blocked
    vram_rows = [r for r in rc.rows if r.name == "VRAM"]
    assert vram_rows and vram_rows[0].status == ResourceStatus.not_enough


def test_help_topics_exist():
    topics = list_topics()
    assert len(topics) >= 10
    body = get_help("filters")
    assert "Denoise" in body or "denoise" in body.lower()
    pl = get_help("filters", "pl")
    assert len(pl) > 10
