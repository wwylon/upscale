# Upscale — developer notes

## Process (Definition of Done)

The canonical local gate is the `verify` list in `publish.manifest.yaml`. Run it before calling a phase done, and keep `docs/help` updated (EN + PL where user-facing). Commit (`phase-N: …`) before the next phase.

```text
uv sync --locked --extra dev --extra server --extra mcp
uv run ruff check upscaler tests
uv run mypy --follow-imports=silent upscaler/schema.py upscaler/job_prep.py upscaler/config.py upscaler/naming.py upscaler/api
uv run upscale --help
uv run pytest -q
uv run python scripts/cpu_smoke.py
```

## Layout

- `upscaler/` — engine, schema, CLI, server, MCP (engine must not import UI)
- `ui/` — static web app
- `desktop/` — pywebview launcher + installers
- `android/` — Kotlin + NCNN
- `docs/help/` — canonical help for web, desktop, Android, MCP
- `locales/` — web UI string catalogues (`en.json` source, `pl.json`); see Localization below


**QA marker locale (`en-XX`).** `/en-XX/` (or `?lang=en-XX`) shows every catalogued string as `X`s, so
any readable English left on screen is hardcoded or missing a key. It is reachable
only from the URL, is not in the language picker, and is never saved as your
choice (it stays for the current tab; picking a language clears it). Regenerate
`locales/en-XX.json` after adding keys: `llm-localize pseudo upscale --locale en-XX`.

## Localization (web UI)

All four pages (`index`, `simple`, `library`, `cli`) and `ui/bridge/*` / `backend-bridge.js` read their
user-visible text from one flat catalogue: `locales/en.json` (English source, key → ICU message) and
`locales/pl.json`, served at `/api/locales/{code}`. The runtime is `ui/i18n.js`, loaded before `support.js`.

- JS: `i18n.t('area.key')` / `i18n.t('area.key', {n, name})` — literal key, ICU placeholders and plurals
  (`{n, plural, one {# image} other {# images}}`), never string concatenation. `this.L(key)` in
  `index.html` is a thin alias.
- Templates: keep the English text in the markup and add `data-i18n="key"` (or
  `data-i18n-title` / `-placeholder` / `-aria-label` / `-alt`). Wrap text that sits next to a
  `{{ binding }}` in its own `<span data-i18n>`; non-English locales patch keyed nodes as React inserts them.
- Locale: `/{code}/page.html` prefix, then `?lang=`, cookie / localStorage `upscale_lang`. The picker
  (Settings on Advanced, header select elsewhere) calls `i18n.setLocale(code)`, which reloads.
- Stay literal: model ids, formats/codecs, CLI flags and commands, units, language endonyms.
- New Polish comes from llm-localize (`.llm-localize.toml` consumer), not hand edits.
  `tests/test_locales.py` checks every key used under `ui/` exists in `en.json`, `pl.json` ⊆ `en.json`,
  and that inline English equals the catalogue.

## Run

Use the `verify` gate above (`uv sync --locked --extra dev --extra server --extra mcp`, then ruff, mypy, `upscale --help`, pytest, and `scripts/cpu_smoke.py`). `pip install -e ".[dev,server,mcp]"` still installs the same extras when uv is not in use.

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
