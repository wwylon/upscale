# Upscale CLI

```
upscale upscale INPUT OUTPUT [options]
upscale batch INPUT_DIR OUTPUT_DIR [options]
upscale compare INPUT OUTPUT_DIR --crop x,y,w,h --models m1,m2
upscale models list|add|verify
upscale serve [--host 127.0.0.1] [--port 8787]
upscale mcp
```

## Common options

| Flag | Default | Notes |
| --- | --- | --- |
| `--scale`, `-s` | — | Mutually exclusive with `--target-size` |
| `--target-size`, `-t` | — | Long edge px |
| `--model`, `-m` | RealESRGAN_x4plus | Built-in name or path |
| `--crop` | — | `x,y,w,h` |
| `--strength` | 1.0 | Blend with Lanczos |
| `--denoise` | 0.5 | wdn model |
| `--blend-model`, `--blend` | — | Second model + weight |
| `--format` | auto | png, jpeg, webp, tiff |
| `--quality` | 90 | JPEG/WebP |
| `--name-pattern` | none (single) / pattern (batch) | Single `upscale` writes exactly to OUTPUT; batch defaults to `{stem}_{scale}x_{model}.{ext}` |
| `--on-exists` | number | number, overwrite, skip |
| `--pre-denoise`, `--jpeg-clean`, `--edge`, `--grain` | 0 | Filters 0–1 |
| `--seamless` | off | Texture wrap |
| `--tile` | auto | |
| `--device` | auto | auto, cuda, cpu, mps |
| `--half` | off | fp16 CUDA |
