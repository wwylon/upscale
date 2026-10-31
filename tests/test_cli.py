from __future__ import annotations

from click.testing import CliRunner

from upscaler.cli import cli, settings_to_cli
from upscaler.config import UpscaleConfig


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
    from upscaler.schema import NamePattern, OnExists, VectorizeSettings

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
    assert "--name-pattern {stem}.{ext}" in s
    assert "--on-exists overwrite" in s
