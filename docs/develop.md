# Upscale — developer notes

## Process (Definition of Done)

Every phase ends with:

1. Tests green (`pytest`)
2. Docs / `docs/help` updated (EN + PL where user-facing)
3. Git commit (`phase-N: …`) before the next phase

## Layout

- `upscaler/` — engine, schema, CLI, server, MCP (engine must not import UI)
- `ui/` — static web app
- `desktop/` — pywebview launcher + installers
- `android/` — Kotlin + NCNN
- `docs/help/` — canonical help for web, desktop, Android, MCP
- `locales/` — UI chrome strings only

## Run

```bash
pip install -e ".[dev,server]"
pytest
python -m upscaler --help
```

## UI entry

- **Preferred:** `upscale serve` (design UI in `ui/` + FastAPI).
- **Legacy:** `upscale ui` (Gradio) — subset of options; do not extend.

## Packaging / smoke

See [plan/design/PACKAGING.md](../plan/design/PACKAGING.md) and [desktop/SMOKE.md](../desktop/SMOKE.md).

## Serve security

Default bind is `127.0.0.1`. Binding a non-loopback host requires `UPSCALE_ALLOW_REMOTE=1` and still exposes unauthenticated job upload, model URL fetch, and job files — trusted LAN only; prefer a reverse proxy with auth for anything else.

Model registry names are confined (`A-Za-z0-9._-`); downloads must be `https://` to an allowlisted host. Output `name_pattern` values are basename-sanitized under the output directory.

## Torch vs ONNX

| Feature | Torch | ONNX (slim) |
|---------|-------|-------------|
| Auto tile | VRAM heuristic | RAM heuristic |
| DNI / blend models | Yes | Ignored (warn) |
| SR alpha | Opt-in `--sr-alpha` | Lanczos alpha only |
| Checkpoint resume | Yes (explicit flag) | Yes |

## MCP subset

MCP stdio tools cover upscale/batch/compare models/list/add/system/help/cancel. Filters, crop, presets, estimate, zip, and full job SSE stay on REST — intentional, not a parity bug.

