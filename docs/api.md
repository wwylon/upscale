# HTTP & MCP API

## FastAPI server

Start with `upscale serve` (default `http://127.0.0.1:8787`). Prefer this over legacy `upscale ui` (Gradio).

Bind is loopback by default. Non-loopback `--host` requires `UPSCALE_ALLOW_REMOTE=1` (API is unauthenticated). Prefer a reverse proxy with auth if you ever bind beyond localhost.

**Jobs directory:** `%LOCALAPPDATA%/Upscale/jobs` (or `$XDG_DATA_HOME/Upscale/jobs`), overridable via `UPSCALE_JOBS_DIR`.

**SSE** (`/events`) is session-scoped to the in-memory job worker — reconnect after process restart returns 404; use `GET /api/jobs/interrupted` + `resume-checkpoint` for strip resume.

**Zip** resolves results from disk via the job id even after in-memory eviction (as long as `result.*` remains under the jobs root).

**Resume-checkpoint** sets an explicit `resume_from_checkpoint` flag so strip tiling reopens the output mmap with `r+` instead of truncating.

**Torch vs ONNX:** ONNX/slim path has auto tile (RAM heuristic), no DNI denoise blend, no second-model blend, Lanczos alpha only.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Redirect to `/{locale}/` (cookie / Accept-Language / `en`) |
| GET | `/{locale}/` | Web UI simple view for an enabled locale (`en`, `pl`) |
| GET | `/{locale}/{page}.html` | Static UI pages (`index`, `simple`, `library`, `cli`) |
| GET | `/{page}.html` | Legacy redirect to `/{locale}/{page}.html` |
| POST | `/api/jobs` | Upload image + JSON `settings`, returns `{id}` |
| GET | `/api/jobs/{id}` | Job state / last pct / CLI hint |
| GET | `/api/jobs/{id}/events` | SSE progress stream |
| POST | `/api/jobs/{id}/cancel` | Cooperative cancel |
| POST | `/api/jobs/{id}/pause` | Pause between tiles |
| POST | `/api/jobs/{id}/resume` | Resume after pause (`type: resumed` event) |
| GET | `/api/jobs/interrupted` | Jobs with on-disk strip checkpoints |
| POST | `/api/jobs/{id}/resume-checkpoint` | Re-queue cancelled/error job from strip flush_row |
| GET | `/api/jobs/{id}/result` | Output file |
| GET | `/api/jobs/{id}/preview` | Input thumbnail |
| GET | `/api/jobs/{id}/display` | Result preview ≤4096 edge |
| GET | `/api/jobs/{id}/passes/{n}` | Mid-pass preview |
| POST | `/api/estimate` | Time / VRAM fit estimate |
| GET | `/api/models` | Model catalog |
| POST | `/api/models` | Add external model (name `^[A-Za-z0-9._-]+$`; https allowlisted hosts only) |
| POST | `/api/models/upload` | Multipart weights upload + register |
| POST | `/api/models/{name}/download` | Download built-in weights |
| GET | `/api/algorithms` | Classical / plugin algorithms (`algorithms` names + `items` metadata) |
| GET | `/api/system` | Device / VRAM / auto tile |
| GET | `/api/about` | App / models dir summary |
| GET | `/api/license` | MIT text |
| GET | `/api/updates` | Optional update check |
| GET | `/api/help` | Help topic index |
| GET | `/api/help/{topic}` | Help markdown (path-confined) |
| GET | `/api/locales/{code}` | UI locale JSON (path-confined) |
| GET | `/api/presets` | Target & option presets |
| POST | `/api/presets/apply` | Resolve preset → settings fragment |
| POST | `/api/suggest` | Model suggestion from upload |
| POST | `/api/benchmark` | Tiny-crop timing (path must be under jobs temp) |
| POST | `/api/compare` | Start crop compare (returns `{id}` immediately) |
| GET | `/api/compare/{id}/status` | Compare state / results (`running`/`done`/`cancelled`) |
| POST | `/api/compare/{id}/cancel` | Cancel an in-flight compare |
| GET | `/api/compare/{id}/{model}` | Compare result image |
| POST | `/api/zip` | Zip completed job results (`job_ids`) |
| GET | `/mcp` | MCP tool **catalog** only (not streamable HTTP MCP) |

Job `settings` follow `JobSettings`. `name_pattern` defaults to `null` so the server writes `result{ext}` unless a pattern is set.

Limits (env override): `UPSCALE_MAX_UPLOAD_BYTES`, `UPSCALE_MAX_QUEUE`, `UPSCALE_MAX_JOBS`, `UPSCALE_JOB_TTL_S`.

## MCP (stdio)

```bash
pip install 'upscaler[mcp]'
upscale mcp
```

Tools: `upscale_image`, `batch_upscale`, `compare_models`, `list_models`, `add_model`, `system_info`, `get_help`, `cancel_job`.

This is a **subset** of the HTTP job API (scale/model/device/half primarily). Full filters/crop/format live on REST. `GET /mcp` is a JSON catalog for discovery, not an HTTP MCP transport.

Long-running MCP tools return while running; `cancel_job` only helps mid-flight before the tool returns.
