"""Classical resize backends — OpenCV when available, else Pillow."""

from __future__ import annotations

import numpy as np
from PIL import Image

BUILTIN_ALGORITHMS = (
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos2",
    "lanczos3",
)

# OpenCV and Pillow only expose one Lanczos kernel (≈3–4 lobe). lanczos2 is
# an alias of lanczos3 so the public names do not promise a difference.
_CV_MAP = {
    "nearest": "INTER_NEAREST",
    "bilinear": "INTER_LINEAR",
    "bicubic": "INTER_CUBIC",
    "lanczos2": "INTER_LANCZOS4",
    "lanczos3": "INTER_LANCZOS4",
}

_PIL_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos2": Image.Resampling.LANCZOS,
    "lanczos3": Image.Resampling.LANCZOS,
}


def _try_cv2():
    try:
        import cv2  # noqa: F401

        return cv2
    except ImportError:
        return None


def resize_classical(name: str, img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize with a classical interpolator. Prefers OpenCV (C++/SIMD)."""
    key = name.strip().lower().replace("-", "_")
    if key not in BUILTIN_ALGORITHMS:
        raise ValueError(f"not a built-in classical algorithm: {name}")

    cv2 = _try_cv2()
    if cv2 is not None:
        return _resize_cv2(cv2, key, img, target_w, target_h)
    return _resize_pil(key, img, target_w, target_h)


def _resize_cv2(cv2, key: str, img: np.ndarray, tw: int, th: int) -> np.ndarray:
    flag = getattr(cv2, _CV_MAP[key])
    # Optional CUDA path when a CUDA build of OpenCV is present
    if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(img)
            out = cv2.cuda.resize(gpu, (tw, th), interpolation=flag)
            return out.download()
        except Exception:  # noqa: BLE001
            pass
    return cv2.resize(img, (tw, th), interpolation=flag)


def _resize_pil(key: str, img: np.ndarray, tw: int, th: int) -> np.ndarray:
    resample = _PIL_MAP[key]
    if img.ndim == 2:
        pil = Image.fromarray(img, mode="L")
        return np.array(pil.resize((tw, th), resample), dtype=np.uint8)
    if img.shape[2] == 1:
        return _resize_pil(key, img[:, :, 0], tw, th)[:, :, None]
    pil = Image.fromarray(img)
    return np.array(pil.resize((tw, th), resample), dtype=np.uint8)


def backend_name() -> str:
    return "opencv" if _try_cv2() is not None else "pillow"
