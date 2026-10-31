# Desktop smoke checklist (macOS — postponed)

- [ ] Build on macOS with `pip install -e ".[desktop,server]"`
- [ ] `python3 desktop/app.py` opens pywebview to `127.0.0.1:8787`
- [ ] Gatekeeper: ad-hoc sign or Developer ID sign the `.app` bundle
- [ ] PyInstaller via `desktop/upscale-macos.spec`
- [ ] MPS device smoke if Apple GPU available

Windows remains the primary packaged target until notarization is automated.
