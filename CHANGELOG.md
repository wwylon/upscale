# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.1] - 2026-11-11

### Fixed

- Windows desktop app crashed on startup (`signal only works in main thread`); the server now registers Ctrl+C handlers only on the main thread.
- README no longer implies `upscale[slim]` removes PyTorch; the PyTorch-free build is the Windows desktop app.
- Build scripts no longer run the design-sync step that overwrote `ui/`; `scripts/sync_design_ui.py` now requires `--force`.


## [0.5.0] - 2026-11-11

### Changed

- **Renamed the Python package from `upscaler` to `upscale`** (import name, distribution name and command now match the repo). Use `import upscale` / `python -m upscale`.
- `LICENSE` is now plain MIT; third-party notes moved to `NOTICE`.
- Added package classifiers, homepage, issue/PR templates and Dependabot config; README documents the lighter `slim` install.

## [0.4.0] - 2026-11-09

### Added

- Web UI localization: every string keyed in `locales/`, language picker, Polish, Chinese and Arabic layouts (including right-to-left).
- Landing site (`site/`) deployed with GitHub Pages, `llms.txt`, and a shorter README.
- `uv.lock` for reproducible installs (`uv sync --locked`).

### Fixed

- Library model listing, resume gating, and Test-tab scale controls in the web UI.
- Job-safety and correctness fixes from a full review of the job API and tiling path.

### Changed

- README slimmed to install and first run; CLI and desktop details live in `docs/`.

## [0.3.0] - 2026-10-31

### Added

- CLI and library upscaler with chained Real-ESRGAN passes and feather-blended tiles.
- FastAPI web UI / job API (`upscale serve`), MCP stdio tools, and Windows slim desktop packaging path.
- Docs under `docs/` (API, help EN/PL), locales, and CI (ruff, mypy subset, pytest, CPU smoke).
