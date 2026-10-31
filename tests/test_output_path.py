"""A0: single-file OUTPUT wins; name_pattern opt-in."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner
from PIL import Image

from upscaler.cli import cli
from upscaler.config import UpscaleConfig
from upscaler.engine import execute
from upscaler.schema import JobSettings, NamePattern


def _tiny_png(path: Path) -> Path:
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    return path


def test_job_settings_name_pattern_defaults_none():
    js = JobSettings(scale=4)
    assert js.name_pattern is None


def test_execute_writes_exact_output_without_pattern(tmp_path: Path):
    src = _tiny_png(tmp_path / "in.png")
    out = tmp_path / "exact_out.png"
    cfg = UpscaleConfig(
        input=src,
        output=out,
        scale=2.0,
        algorithm="nearest",
        quiet=True,
        name_pattern=None,
    )
    actual = execute(cfg)
    assert actual == out
    assert out.is_file()


def test_execute_renames_when_pattern_set(tmp_path: Path):
    src = _tiny_png(tmp_path / "photo.png")
    out = tmp_path / "ignored.png"
    cfg = UpscaleConfig(
        input=src,
        output=out,
        scale=2.0,
        algorithm="nearest",
        quiet=True,
        name_pattern=NamePattern(pattern="{stem}_{scale}x_{model}.{ext}"),
    )
    actual = execute(cfg)
    assert actual.name == "photo_2x_nearest.png"
    assert actual.is_file()
    assert actual != out


def test_cli_upscale_default_writes_output_arg(tmp_path: Path):
    src = _tiny_png(tmp_path / "in.png")
    out = tmp_path / "out.png"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["upscale", str(src), str(out), "--scale", "2", "--algorithm", "nearest", "-q"],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
