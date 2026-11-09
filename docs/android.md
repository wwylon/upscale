# Android

See [android/README.md](../android/README.md) for NCNN + remote-engine architecture.

- Minimal Kotlin `MainActivity` placeholder under `android/app/`.
- Help topic ids listed in `android/assets/help/README.md`.
- ONNX export: `py -3 scripts/export_onnx.py 4x-spanx4-ch48`

Production builds will add Gradle, NCNN libs, and optional LAN client for `upscale serve`.
