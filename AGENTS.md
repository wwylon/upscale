# AGENTS.md — upscale (public)

Bootstrap for agents working on the **public** upscale tree.

1. [README.md](README.md) — install, CLI, desktop, library
2. [docs/api.md](docs/api.md) — HTTP / MCP surface
3. [docs/develop.md](docs/develop.md) — contributor-oriented notes (if present)
4. [CONTRIBUTING.md](CONTRIBUTING.md) — tests and PR expectations

## Hard constraints

- Do not invent CLI flags or API routes; check `upscale --help` and `docs/api.md`.
- Model weights under `models/` are runtime downloads — never commit `.pth` / `.onnx` binaries.
- Prefer `upscale serve` over legacy Gradio `upscale ui`.

## Tests

```text
pip install -e ".[dev,server,mcp]"
ruff check upscale tests
pytest -q
python scripts/cpu_smoke.py
```
