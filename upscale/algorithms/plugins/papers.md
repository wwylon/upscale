# Paper / research algorithm candidates

Evaluation status for the paper-algorithms track (Phases 0–2 shipped under
Classical → Paper). Diffusion stacks stay out of the slim installer.

## Shipped

| Id | Status | Notes |
| --- | --- | --- |
| **texture_residual** | shipped | Lanczos structure + LR high-pass residual. Inspired by structure/texture split (Grimm/Kulla NPR shading, Efros–Leung / quilting family). **Not** a port of [pg2003uts](https://web.engr.oregonstate.edu/~grimmc/content/papers/pg2003uts/pg2003uts.pdf) (NPR mesh shading, not SR). |
| **vectorize_scale** | shipped (optional) | Engines: [VTracer](https://pypi.org/project/vtracer/) (default) or [PyAutoTrace](https://pypi.org/project/pyautotrace/) (classic AutoTrace). Raster→SVG, then [resvg_py](https://pypi.org/project/resvg_py/) → PNG. Extra: `pip install 'upscale[vectorize]'`. Best for logos / line art; weak on photos. SVG is internal only — user upload of SVG remains blocked. |

## Icebox / watch (Torch extras later)

| Candidate | Why deferred |
| --- | --- |
| **DiWa** ([brian-moser/diwa](https://github.com/brian-moser/diwa), arXiv 2304.01994) | Diffusion in wavelet domain; needs Torch + trained weights. Optional research extra only. |
| **TOC-SR** (arXiv 2605.02767) | Compact one-step diffusion SR; custom weights / VAE. Watch for a slim Torch extra later. |

## Out of scope

| Candidate | Why |
| --- | --- |
| **HiWave** (alphaXiv 2506.20452) | Training-free ultra-HR from SDXL — generative foundation stack, minutes per 4K. |
| **Pixelsmith** ([Thanos-DB/Pixelsmith](https://github.com/Thanos-DB/Pixelsmith)) | Same class: foundation-model generation, not classical resize. |
| **TGSR** (arXiv 2610.14281) | Test-time text control for generative SR — needs a diffusion SR backbone. |

## Earlier pointers (still valid)

1. **Image Analogies** (Hertzmann et al.) — style / texture transfer by analogy.
2. **Texler** / related texture synthesis — example-based texture expand.
3. **Classical texture synth** (Efros–Leung / quilting) — non-neural fill (partially adapted in `texture_residual`).

License-review any external weights or reference code before shipping.

## Native kernels (later)

C++ / CUDA / Vulkan under `plugins/native/` remains a stub. Prefer NumPy /
OpenCV / Pillow for paper plugins unless a method truly needs a custom kernel.
Do not vendor diffusion stacks into the slim Windows installer.
