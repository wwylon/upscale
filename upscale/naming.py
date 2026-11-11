"""Resolve output paths from NamePattern with collision handling."""

from __future__ import annotations

from pathlib import Path

from .schema import NamePattern, OnExists


def resolve_output_path(
    directory: Path,
    *,
    stem: str,
    scale: float | str,
    model: str,
    ext: str,
    pattern: NamePattern | None = None,
    width: int | None = None,
    height: int | None = None,
) -> Path | None:
    """
    Build an output path under directory.
    Returns None if on_exists=skip and the file already exists.
    """
    root = directory.resolve()
    np = pattern or NamePattern()
    name = np.render(
        stem=stem, scale=scale, model=model, ext=ext, w=width, h=height, n=None
    )
    path = (directory / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Output path escapes directory: {path}")

    if not path.exists():
        return path

    if np.on_exists == OnExists.overwrite:
        return path
    if np.on_exists == OnExists.skip:
        return None

    # number: stem_1.ext, stem_2.ext, …
    for i in range(1, 10_000):
        numbered = np.render(
            stem=stem,
            scale=scale,
            model=model,
            ext=ext,
            w=width,
            h=height,
            n=i,
        )
        # If pattern has no {n}, insert before extension
        if "{n}" not in np.pattern:
            candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        else:
            candidate = directory / numbered
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Output path escapes directory: {candidate}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find free name under {directory}")
