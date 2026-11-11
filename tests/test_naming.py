from __future__ import annotations

from pathlib import Path

from upscale.naming import resolve_output_path
from upscale.schema import NamePattern, OnExists


def test_resolve_new_file(tmp_path: Path):
    p = resolve_output_path(
        tmp_path, stem="a", scale=4, model="m", ext="png"
    )
    assert p is not None
    assert p.name == "a_4x_m.png"


def test_number_on_collision(tmp_path: Path):
    (tmp_path / "a_4x_m.png").write_bytes(b"x")
    p = resolve_output_path(
        tmp_path,
        stem="a",
        scale=4,
        model="m",
        ext="png",
        pattern=NamePattern(on_exists=OnExists.number),
    )
    assert p is not None
    assert p.name == "a_4x_m_1.png"


def test_skip_on_collision(tmp_path: Path):
    (tmp_path / "a_4x_m.png").write_bytes(b"x")
    p = resolve_output_path(
        tmp_path,
        stem="a",
        scale=4,
        model="m",
        ext="png",
        pattern=NamePattern(on_exists=OnExists.skip),
    )
    assert p is None


def test_name_pattern_rejects_path_escape(tmp_path: Path):
    p = resolve_output_path(
        tmp_path,
        stem="../evil",
        scale=4,
        model="m",
        ext="png",
        pattern=NamePattern(pattern="{stem}_{scale}x_{model}.{ext}"),
    )
    assert p is not None
    assert p.parent.resolve() == tmp_path.resolve()
    assert ".." not in p.name
    assert p.name.startswith("evil_") or "_evil" in p.name or p.name.startswith("_")
