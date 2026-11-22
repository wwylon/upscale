# Backends

Upscale runs neural models on **Torch** (default when PyTorch is installed) or **ONNX Runtime** (slim desktop / `UPSCALE_BACKEND=onnx`). Feature support differs:

| Feature | Torch | ONNX | Notes |
| --- | --- | --- | --- |
| Neural RGB upscale | Yes | Yes | Same tiling / feather path |
| `--sr-alpha` | Yes | No (Lanczos only) | Soft alpha is usually better with Lanczos; SR alpha is Torch-only until ONNX gains a matching path |
| `--denoise` DNI | Continuous weight blend for `realesr-general-x4v3` / WDN | Soft swap: if denoise &gt; 0.5 and WDN `.onnx` exists, uses WDN; otherwise ignored with a warning | Prefer Torch for intermediate denoise values |
| `--blend-model` | Weight interpolation | Ignored (warning) | Export a pre-blended ONNX or use Torch |
| FP16 / `--half` | CUDA only; MPS stays fp32 | N/A (provider-dependent) | Half is ignored on CPU |

Classical algorithms and paper plugins do not need a neural backend.

Set the backend with `UPSCALE_BACKEND=torch|onnx|auto` (auto picks Torch when importable).
