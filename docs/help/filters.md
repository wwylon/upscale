# Filters

Optional steps with amounts from 0–100%, available on the **Scale** tab:

- **Before:** Denoise, JPEG artifact removal, Deblur
- **After:** Edge enhance, Halo reduce, Color and contrast, Film grain

**Reset filters** clears all. Filters apply to Scale runs (full image or a single test crop upscale). Test **compare** does not run filter variants — use Scale filters after you pick a model. CLI: `--pre-denoise`, `--jpeg-clean`, `--deblur`, `--edge`, `--halo-reduce`, `--color-contrast`, `--grain`.
