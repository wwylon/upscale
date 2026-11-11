from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from upscale.cli import cli, settings_to_cli
from upscale.config import UpscaleConfig


def test_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.output
    assert "compare" in result.output


def test_models_list():
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0
    assert "RealESRGAN_x4plus" in result.output


def test_settings_to_cli():
    cfg = UpscaleConfig(input=None, output=None, scale=4, model="m", strength=0.5)
    s = settings_to_cli(cfg)
    assert "--scale 4" in s
    assert "--strength 0.5" in s


def test_settings_to_cli_overlap_tiff_vectorize_on_exists():
    from upscale.schema import NamePattern, OnExists, VectorizeSettings

    cfg = UpscaleConfig(
        input=None,
        output=None,
        scale=2,
        model="m",
        overlap=64,
        tiff_compression="deflate",
        algorithm="vectorize_scale",
        vectorize=VectorizeSettings(engine="autotrace", color_precision=4, mode="polygon"),
        name_pattern=NamePattern(pattern="{stem}.{ext}", on_exists=OnExists.overwrite),
    )
    s = settings_to_cli(cfg)
    assert "--overlap 64" in s
    assert "--tiff-compression deflate" in s
    assert "--vectorize-engine autotrace" in s
    assert "--vectorize-color-precision 4" in s
    assert "--vectorize-mode polygon" in s
    assert "--name-pattern" in s
    assert "{stem}.{ext}" in s
    assert "--on-exists overwrite" in s


def test_validate_sizing_rejects_height_only():
    import click

    from upscale.job_prep import validate_sizing

    cfg = UpscaleConfig(height=100)
    with pytest.raises(click.UsageError, match="width and height"):
        validate_sizing(cfg)


def test_batch_continues_after_one_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src_dir = tmp_path / "in"
    dst_dir = tmp_path / "out"
    src_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(src_dir / "a.png")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(src_dir / "b.png")

    from upscale import cli as cli_mod

    calls = {"n": 0}
    real_execute = cli_mod.execute

    def flaky(cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real_execute(cfg)

    monkeypatch.setattr(cli_mod, "execute", flaky)
    result = CliRunner().invoke(
        cli,
        ["batch", str(src_dir), str(dst_dir), "--algorithm", "nearest", "--scale", "2", "-q"],
    )
    assert calls["n"] == 2
    assert result.exit_code != 0
    assert "boom" in result.output
