"""Shared neural weight blending for DNI denoise and --blend-model."""

from __future__ import annotations

import gc
import warnings
from typing import Any, Callable

import numpy as np


def blend_parameters(dst_model: Any, src_model: Any, alpha: float) -> None:
    """Atomically blend ``src`` into ``dst``: dst ← (1-α)·dst + α·src.

    Builds a full replacement state dict and assigns it only after every
    parameter name and shape matches. Raises on mismatch (caller may warn).
    """
    import torch

    a = float(np.clip(alpha, 0.0, 1.0))
    dst_sd = {k: v.detach() for k, v in dst_model.state_dict().items()}
    src_sd = {k: v.detach() for k, v in src_model.state_dict().items()}
    if set(dst_sd) != set(src_sd):
        raise ValueError("blend_parameters: state dict keys differ")
    blended = {}
    for k, v_dst in dst_sd.items():
        v_src = src_sd[k]
        if v_dst.shape != v_src.shape:
            raise ValueError(f"blend_parameters: shape mismatch on {k}")
        if not torch.is_floating_point(v_dst):
            blended[k] = v_dst.clone()
        else:
            blended[k] = v_dst.float().mul(1.0 - a).add_(v_src.float(), alpha=a).to(
                dtype=v_dst.dtype
            )
    dst_model.load_state_dict(blended, strict=True)


def blend_secondary_into(
    descriptor: Any,
    *,
    other_name: str,
    alpha: float,
    weights_dir: Any,
    device: Any,
    half: bool,
    ensure_model: Callable,
    load_model: Callable,
    log: Callable[[str], None] | None = None,
    show: bool = False,
    label: str = "blend",
) -> None:
    """Load ``other_name``, blend into ``descriptor.model`` with weight ``alpha``, restore device."""
    import torch

    a = float(np.clip(alpha, 0.0, 1.0))
    try:
        other_path = ensure_model(other_name, weights_dir)
        other = load_model(other_path, device, half)
        other.model.to("cpu")
        descriptor.model.to("cpu")
        blend_parameters(descriptor.model, other.model, a)
        descriptor.model.to(device)
        if show and log is not None:
            log(f"{label}: blended with {other_name} @ {a:.2f}")
        del other
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"{label} skipped: {exc}", stacklevel=2)
        try:
            descriptor.model.to(device)
        except Exception:  # noqa: BLE001
            pass
