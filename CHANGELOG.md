# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
