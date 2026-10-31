# Android

See [android/README.md](../android/README.md) for NCNN + remote-engine architecture.

- Minimal Kotlin `MainActivity` placeholder under `android/app/`.
- Help topic ids listed in `android/assets/help/README.md`.
- ONNX export stub: `py -3 scripts/export_onnx_stub.py models/RealESRGAN_x4plus.pth`

Production builds will add Gradle, NCNN libs, and optional LAN client for `upscale serve`.
