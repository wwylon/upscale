# Shortcuts

| Action | CLI |
|--------|-----|
| Single file | `upscale upscale in.png out.png -s 4` |
| Batch folder | `upscale batch ./in ./out -s 4` |
| Fast SPAN | add `--fast` or preset `fast_4x` |
| Target 4K long edge | `--target-size 3840` or preset `4k` |
| Model hint | `upscale suggest in.png` |
| Timing sample | `upscale benchmark in.png --device cpu` |
| Web UI | `upscale serve` |
| Agents | `upscale mcp` |
