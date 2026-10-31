#!/usr/bin/env python
"""Export a registry model to ONNX for the slim (Torch-free) desktop backend.

Requires Torch + spandrel on the *build* machine only::

    py -3 scripts/export_onnx.py 4x-spanx4-ch48 -o models/4x-spanx4-ch48.onnx

The Windows slim app loads the .onnx with ONNX Runtime (no PyTorch).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SR weights to ONNX")
    parser.add_argument(
        "model",
        help="Registry name (e.g. 4x-spanx4-ch48) or path to .pth/.safetensors",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--size", type=int, default=64, help="Dummy trace spatial size")
    args = parser.parse_args()

    import torch

    from upscaler.models import DEFAULT_WEIGHTS_DIR, ensure_model, load_model, registry_scale

    name = args.model
    weights = Path(name)
    if weights.is_file():
        model_path = weights
        scale = 4
        out_default = weights.with_suffix(".onnx")
    else:
        model_path = ensure_model(name, DEFAULT_WEIGHTS_DIR)
        scale = registry_scale(name)
        out_default = DEFAULT_WEIGHTS_DIR / f"{name}.onnx"

    out = args.output or out_default
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    descriptor = load_model(model_path, device, fp16=False)
    model = descriptor.model.eval()
    # Prefer scale from descriptor when available
    scale = int(getattr(descriptor, "scale", scale) or scale)

    dummy = torch.randn(1, 3, args.size, args.size, device=device)
    print(f"Exporting {model_path} → {out} (scale≈{scale}, opset={args.opset})")
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "n", 2: "h", 3: "w"},
            "output": {0: "n", 2: "h", 3: "w"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
