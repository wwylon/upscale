# Desktop smoke checklist (Windows)

## Dev mode (Python)

- [ ] `pip install -e ".[desktop,server]"` + `onnxruntime`
- [ ] `py -3 scripts/export_onnx.py 4x-spanx4-ch48`
- [ ] `$env:UPSCALE_BACKEND='onnx'; py -3 desktop/app.py` opens design UI
- [ ] Upscale a small PNG; before/after shows result

## Release slim build (ship this)

- [ ] `.\desktop\build_windows_slim.ps1`
- [ ] `dist\Upscale\` has **no** `torch*.dll`
- [ ] Onedir size well under ~400 MB (ideally &lt; 150 MB)
- [ ] `Upscale-Setup-slim.exe` or portable zip installs/runs without Python
- [ ] Copy `4x-spanx4-ch48.onnx` into `%LOCALAPPDATA%\Upscale\models`
- [ ] Upscale works on CPU (and DirectML if ORT reports it)

## Do not ship

- Torch/CUDA PyInstaller builds from `build_windows.ps1` (multi‑GB) — local experiment only.
