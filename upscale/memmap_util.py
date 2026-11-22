"""Shared memmap detection, materialization, and row-chunked spill helpers."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np


def is_memmap(x: np.ndarray) -> bool:
    """True for numpy memmap arrays (including monkeypatched test doubles)."""
    mm = getattr(np, "memmap", None)
    if isinstance(mm, type):
        try:
            if isinstance(x, mm):
                return True
        except TypeError:
            pass
    return getattr(x, "filename", None) is not None or type(x).__name__ == "memmap"


def materialize(arr: np.ndarray) -> np.ndarray:
    """Return a plain ndarray copy when ``arr`` is memmap-backed."""
    if is_memmap(arr):
        return np.array(arr, copy=True)
    return np.asarray(arr)


def spill_uint8_rows(
    arr: np.ndarray,
    dest: Path | str | BinaryIO,
    *,
    chunk: int = 64,
) -> None:
    """Write ``arr`` as contiguous uint8, ``chunk`` rows at a time.

    ``dest`` may be a filesystem path (opened ``wb``) or an open binary file.
    """
    src = np.ascontiguousarray(arr, dtype=np.uint8)
    step = max(1, int(chunk))
    own = False
    if hasattr(dest, "write"):
        f = dest  # type: ignore[assignment]
    else:
        f = open(dest, "wb")  # noqa: SIM115 — closed below
        own = True
    try:
        for y0 in range(0, src.shape[0], step):
            y1 = min(src.shape[0], y0 + step)
            f.write(src[y0:y1].tobytes())
    finally:
        if own:
            f.close()


def close_memmap(arr: np.ndarray | None) -> None:
    """Best-effort close for Windows file locks on memmap handles."""
    if arr is None:
        return
    try:
        if hasattr(arr, "flush"):
            arr.flush()
    except Exception:  # noqa: BLE001
        pass
    try:
        if hasattr(arr, "_mmap") and arr._mmap is not None:
            arr._mmap.close()
    except Exception:  # noqa: BLE001
        pass
