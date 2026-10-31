# Paper-algorithm plugins

Modules in this package are auto-imported by `load_plugins()`. Each module
should call `register_algorithm(...)` so the name appears under Classical →
Paper in the UI and in `GET /api/algorithms`.

## Shipped

| Name | Extra | Role |
| --- | --- | --- |
| `texture_residual` | none | Lanczos structure + high-frequency residual |
| `vectorize_scale` | `upscaler[vectorize]` (`vtracer` + `pyautotrace` + `resvg_py`) | SVG trace (VTracer or AutoTrace), then rasterize at target size |

See [papers.md](papers.md) for evaluation (DiWa / TOC-SR icebox; HiWave family out of scope).

## Hook

```python
from upscaler.algorithms import register_algorithm

def my_resize(img, target_w, target_h):
    ...  # HWC uint8 RGB → same

register_algorithm(
    "my_method",
    my_resize,
    label="My method",
    group="paper",
    description="One-line help for the UI.",
)
```

Then set `JobSettings.algorithm = "my_method"` (or CLI `--algorithm`).

## Native kernels (later)

| Path | When |
| --- | --- |
| OpenCV / Pillow | Classical + paper plugins now |
| C++ + pybind11 | CPU paper kernels |
| CUDA extension | NVIDIA |
| Vulkan compute | Cross-vendor GPU (aligns with NCNN packaging) |

Do not vendor large native toolchains or diffusion stacks into the slim
Windows installer until a specific method is scheduled as a research extra.
