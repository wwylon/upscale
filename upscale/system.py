"""Device / RAM / disk probes for preflight resource checks."""

from __future__ import annotations

import os
from pathlib import Path

from .models import DEFAULT_WEIGHTS_DIR
from .schema import SystemInfo


def gather_system_info(
    *,
    device: str = "auto",
    fp16: bool = False,
    weights_dir: Path | None = None,
) -> SystemInfo:
    cuda = False
    mps = False
    gpu_name = None
    vram_total = vram_free = None
    auto_tile = 256
    resolved = "cpu"

    try:
        import torch

        from .models import auto_tile_size, resolve_device

        dev = resolve_device(device)
        resolved = str(dev)
        cuda = torch.cuda.is_available()
        mps = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        if getattr(dev, "type", None) == "cuda":
            gpu_name = torch.cuda.get_device_name(dev)
            free_b, total_b = torch.cuda.mem_get_info(dev)
            vram_total = int(total_b // (1024 * 1024))
            vram_free = int(free_b // (1024 * 1024))
        auto_tile = auto_tile_size(dev, fp16)
    except ImportError:
        # Slim / ONNX desktop — report ORT providers instead of Torch
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "DmlExecutionProvider" in providers:
                resolved = "onnx/directml"
                gpu_name = "DirectML"
            elif "CUDAExecutionProvider" in providers:
                resolved = "onnx/cuda"
                gpu_name = "CUDA (ORT)"
            else:
                resolved = "onnx/cpu"
        except ImportError:
            resolved = "cpu"

    ram_total = ram_avail = None
    try:
        import psutil  # optional

        vm = psutil.virtual_memory()
        ram_total = int(vm.total // (1024 * 1024))
        ram_avail = int(vm.available // (1024 * 1024))
    except ImportError:
        ram_avail = None

    disk_free = None
    root = weights_dir or Path(os.environ.get("UPSCALE_MODELS_DIR", DEFAULT_WEIGHTS_DIR))
    try:
        usage = os.statvfs(str(root)) if hasattr(os, "statvfs") else None
        if usage is not None:
            disk_free = int(usage.f_bavail * usage.f_frsize // (1024 * 1024))
        else:
            import shutil

            disk_free = int(shutil.disk_usage(str(Path(root).anchor or ".")).free // (1024 * 1024))
    except OSError:
        disk_free = None

    try:
        from .backends import resolve_backend

        backend = resolve_backend()
    except Exception:  # noqa: BLE001
        backend = "torch"

    return SystemInfo(
        device=resolved,
        cuda_available=cuda,
        mps_available=mps,
        gpu_name=gpu_name,
        vram_total_mb=vram_total,
        vram_free_mb=vram_free,
        ram_total_mb=ram_total,
        ram_available_mb=ram_avail,
        disk_free_mb=disk_free,
        auto_tile=auto_tile,
        models_dir=str(root),
        backend=backend,
    )
