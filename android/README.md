# Android (skeleton)

Planned architecture:

1. **On-device NCNN** — export ONNX from spandrel weights, convert with NCNN tools, run tiled inference in Kotlin/C++.
2. **Remote engine** — optional client for `upscale serve` on LAN (upload job, SSE progress, download result).

This repo ships structure and docs only; full NCNN pipeline is not wired yet.

## Layout

```
android/
  app/src/main/java/com/upscale/app/MainActivity.kt
  assets/help/README.md          # bundled help topic ids
docs/android.md
scripts/export_onnx_stub.py
android/export_ncnn.md
```

## Next steps

- Implement `MainActivity` WebView or Compose UI calling remote API.
- Replace `export_onnx_stub.py` with real spandrel → ONNX export per model.
- Add Gradle wrapper and NCNN `.so` per ABI.
