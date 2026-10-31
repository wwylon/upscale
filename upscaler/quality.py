"""Seam and diff helpers for QA (numpy-only)."""

from __future__ import annotations

import numpy as np


def strip_seam_score(left: np.ndarray, right: np.ndarray, overlap: int) -> float:
    """
    Mean absolute difference in the horizontal overlap between two strips.
    Lower is better (less visible seam).
    """
    if overlap <= 0:
        return 0.0
    if left.ndim == 2:
        left = left[..., None]
        right = right[..., None]
    ov = min(overlap, left.shape[1], right.shape[1])
    a = left[:, -ov:, :].astype(np.float32)
    b = right[:, :ov, :].astype(np.float32)
    return float(np.mean(np.abs(a - b)))


def diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """PSNR and mean abs error between two same-shape arrays."""
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    mae = float(np.mean(np.abs(af - bf)))
    mse = float(np.mean((af - bf) ** 2))
    if mse <= 1e-12:
        psnr = 99.0
    else:
        psnr = float(10.0 * np.log10((255.0**2) / mse))
    return {"mae": mae, "mse": mse, "psnr": psnr}
