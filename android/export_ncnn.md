# Export weights to NCNN

1. Export ONNX from PyTorch/spandrel (`scripts/export_onnx_stub.py` today prints steps only).
2. `onnx2ncnn model.onnx model.param model.bin`
3. Optimize: `ncnnoptimize model.param model.bin model-opt.param model-opt.bin 0`
4. Ship `.param`/`.bin` under `android/app/src/main/assets/models/`.

Tile size should match desktop defaults; feather blending stays on CPU in Kotlin for v1.
