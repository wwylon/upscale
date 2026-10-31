# First run

1. Install: `pip install -e ".[server]"` (add `[ui]` or `[desktop]` if needed).
2. Run `upscale models list` — first upscale downloads weights into `models/`.
3. Try `upscale suggest samples/tiny.png` for a model hint.
4. Upscale: `upscale upscale input.png out.png --scale 4 --cpu` on machines without CUDA.
5. Web UI: `upscale serve` → http://127.0.0.1:8787/en/ (opens **Simple** by default; last **Simple** / **Advanced** choice is remembered).

Use `upscale presets list` for FHD/4K targets.
