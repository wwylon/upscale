# upscale

AI image upscaler — chains Real-ESRGAN (via [spandrel](https://github.com/chaiNNer-org/spandrel)) passes to reach arbitrary target scales, with feather-blended overlapping tiles for seam-free output.

## Install

```bash
pip install -e .
pip install -e ".[server]"    # FastAPI UI + job API
pip install -e ".[mcp]"        # Agent MCP stdio tools
pip install -e ".[desktop]"    # pywebview shell (with server)
pip install -e ".[ui]"          # legacy Gradio
pip install -e ".[dev]"         # pytest, httpx
```

## Quick start

```bash
upscale --help
upscale models list
upscale suggest samples/tiny.png
upscale upscale input.png output.png --scale 4 --cpu
upscale serve                   # http://127.0.0.1:8787/en/  (prefer this over Gradio `upscale ui`)
upscale mcp                     # stdio MCP for agents
```

Equivalent module form: `python -m upscaler upscale …`

## CLI overview

| Command | Purpose |
|---------|---------|
| `upscale upscale` | Single image |
| `upscale batch` | Folder of images |
| `upscale compare` | Same crop, multiple models |
| `upscale models` | List / add / verify weights |
| `upscale suggest` | Heuristic model pick |
| `upscale presets list` | FHD / 4K / print + Photo 4× bundles |
| `upscale benchmark` | Tiny-crop timing (saved under user config) |
| `upscale serve` | Web UI + REST API |
| `upscale mcp` | MCP tools over stdio |
| `upscale ui` | Gradio (legacy — use `serve`) |

### Notable flags

| Flag | Meaning |
|------|---------|
| `--scale` / `--target-size` | Exact scale factor, or target long-edge pixels |
| `--model` | Built-in name (auto-download) or path to `.pth` |
| `--fast` | Use SPAN (`4x-spanx4-ch48`) |
| `--sr-alpha` | Upscale alpha with SR (default: Lanczos resize) |
| `--disk-intermediate-threshold` | Spill large mid-pass frames to disk (default 16000; `0` disables) |
| `--tile auto` | Pick tile from free VRAM |
| `--half` | fp16 on CUDA |

**Alpha:** without `--sr-alpha`, transparent images keep RGB SR quality while alpha is Lanczos-resized to the output size.

## Desktop (Windows) — small download (ONNX)

Shippable Windows builds **do not include PyTorch**. The app uses ONNX Runtime; export models once on a Torch machine.

**Dev (Python + Torch):**

```bash
pip install -e ".[desktop,server]"
py -3 scripts/sync_design_ui.py
set UPSCALE_BACKEND=onnx
py -3 scripts/export_onnx.py 4x-spanx4-ch48
py -3 desktop/app.py
```

**Release build (slim, no Torch in the bundle):**

```powershell
.\desktop\build_windows_slim.ps1
```

Produces `desktop\Output\Upscale-Setup-slim.exe` and/or `Upscale-portable-slim.zip` — target tens–low hundreds of MB, not multi‑GB.  
Place `*.onnx` in `%LOCALAPPDATA%\Upscale\models` (or export there).  

`.\desktop\build_windows.ps1` (Torch onedir) is **dev-only** and must not be published as the user download.

macOS packaging is postponed (`desktop/MAC_SMOKE.md`).

## HTTP & MCP

See [docs/api.md](docs/api.md). Help topics live under `docs/help/` (EN + `pl/`). `GET /mcp` lists MCP tool names; run `upscale mcp` for stdio transport.

## Android

Skeleton only — [docs/android.md](docs/android.md), [android/README.md](android/README.md).

## Library

```python
from upscaler import UpscaleConfig, execute, compute_passes, MODELS
from upscaler.suggest import suggest_model
from upscaler.presets import list_presets

cfg = UpscaleConfig(input="in.png", output="out.png", scale=4, cpu=True)
execute(cfg)
```

## Tests

```bash
pip install -e ".[dev,server,mcp]"
pytest
```

## Layout

- `upscaler/cli.py` — Click entry
- `upscaler/engine.py` — shared run path
- `upscaler/server.py` — FastAPI jobs + static UI
- `upscaler/mcp_server.py` — MCP stdio tools
- `upscaler/suggest.py`, `presets.py`, `benchmark.py` — helpers
- `upscaler/checkpoint.py`, `quality.py` — resume/QA foundations
- `upscaler/pipeline.py` — pass loop, disk intermediates
- `upscaler/tiling.py` — feather-blended tiles
- `upscaler/models.py` — registry, download + hash verify
- `upscaler/web.py` — Gradio (legacy)
- `desktop/` — pywebview shell
- `android/` — mobile skeleton
- `models/` — downloaded weights (gitignored)
