# Agents / MCP

Use `upscale mcp` for a **stdio** MCP server aimed at agents. Tools cover a focused subset: single/batch upscale, compare, list/add models, system info, help, and cancel.

`GET /mcp` on the FastAPI server returns a JSON **catalog** of those tools for discovery. It is not a streamable HTTP MCP endpoint.

For full job options (filters, crop, formats, pause/resume, SSE progress), use the REST API (`upscale serve`) documented in `docs/api.md`.

The Windows desktop app (`desktop/app.py`) bundles the same FastAPI UI via pywebview on `127.0.0.1:8787`.
