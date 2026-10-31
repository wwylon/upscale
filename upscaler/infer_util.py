"""Shared patch normalize / denormalize for Torch and ONNX infer wrappers."""

from __future__ import annotations

import numpy as np


def patch_to_nchw_float(patch: np.ndarray) -> np.ndarray:
    """uint8 HWC → float32 NCHW in [0, 1]."""
    t = patch.astype(np.float32) / 255.0
    if t.ndim == 2:
        t = t[:, :, None]
    return np.transpose(t, (2, 0, 1))[None, ...]


def nchw_float_to_uint8(out: np.ndarray) -> np.ndarray:
    """float NCHW or CHW in [0, 1] → uint8 HWC."""
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
