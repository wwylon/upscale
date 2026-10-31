# CLAUDE.md — upscale (public)

Public checkout guidance. Prefer [AGENTS.md](AGENTS.md) and [docs/](docs/).

- Package: `upscaler/` (`cli.py`, `pipeline.py`, `tiling.py`, `models.py`, `server.py`, …)
- Web UI: `ui/` + FastAPI in `upscaler/server.py` / `upscaler/api/`
- Desktop: `desktop/` (slim ONNX path); Android skeleton: `android/`
- Docs hub: [docs/api.md](docs/api.md), help under `docs/help/`

## Run

```text
pip install -e ".[dev,server,mcp]"
upscale --help
```

Do not assume a sibling `devkit` or shared inbox layout; this repository is standalone for public users.
