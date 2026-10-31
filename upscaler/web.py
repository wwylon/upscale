"""Gradio web UI — thin wrapper over UpscaleConfig + engine.execute.

Deprecated for new workflows: prefer ``upscale serve`` (FastAPI + bundled HTML UI).
This module remains for ``upscale ui`` / ``pip install upscaler[ui]`` compatibility.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .config import UpscaleConfig
from .engine import execute
from .models import MODELS


def build_app():
    import gradio as gr

    model_choices = list(MODELS.keys())

    def _run(
        image,
        scale,
        target_size,
        model,
        tile,
        overlap,
        device,
        half,
        fast,
        sr_alpha,
        progress=gr.Progress(track_tqdm=True),
    ):
        if image is None:
            raise gr.Error("Upload an image first.")
        if (scale is None or scale <= 0) and (target_size is None or target_size <= 0):
            raise gr.Error("Set a positive --scale or target long-edge size.")

        progress(0, desc="starting")

        def cb(msg: str, frac: float) -> None:
            progress(frac, desc=msg)

        with tempfile.TemporaryDirectory(prefix="upscale_ui_") as tmp:
            tmp_path = Path(tmp)
            # Gradio may give a numpy array or a path-like
            if hasattr(image, "save"):
                in_path = tmp_path / "input.png"
                image.save(in_path)
            else:
                in_path = Path(image)
            out_path = tmp_path / "output.png"

            use_scale = float(scale) if scale and scale > 0 else None
            use_target = int(target_size) if target_size and target_size > 0 else None
            if use_scale is not None and use_target is not None:
                use_target = None  # prefer scale when both set from UI

            cfg = UpscaleConfig(
                input=in_path,
                output=out_path,
                scale=use_scale,
                target_size=use_target,
                model=model,
                tile="auto" if tile == "auto" else int(tile),
                overlap=int(overlap),
                device=device,
                half=bool(half),
                quiet=True,
                fast=bool(fast),
                sr_alpha=bool(sr_alpha),
            )
            actual = execute(cfg, progress_callback=cb, echo=lambda _m: None)
            from PIL import Image

            return Image.open(actual).copy()

    with gr.Blocks(title="upscale") as demo:
        gr.Markdown("# upscale\nReal-ESRGAN with feather-blended tiles.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input")
                scale = gr.Number(value=4, label="Scale", precision=3)
                target_size = gr.Number(value=0, label="Target long edge (0 = use scale)", precision=0)
                model = gr.Dropdown(choices=model_choices, value="RealESRGAN_x4plus", label="Model")
                tile = gr.Dropdown(
                    choices=["auto", "128", "256", "512", "768"],
                    value="auto",
                    label="Tile",
                )
                overlap = gr.Slider(0, 128, value=32, step=1, label="Overlap")
                device = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Device")
                half = gr.Checkbox(label="fp16 (CUDA)", value=False)
                fast = gr.Checkbox(label="Fast mode (SPAN)", value=False)
                sr_alpha = gr.Checkbox(label="SR alpha channel", value=False)
                btn = gr.Button("Upscale", variant="primary")
            with gr.Column():
                out = gr.Image(type="pil", label="Output")
                gallery = gr.Gallery(label="Before / after", columns=2)

        def _run_and_gallery(*args):
            result = _run(*args)
            src = args[0]
            return result, [src, result]

        btn.click(
            _run_and_gallery,
            inputs=[inp, scale, target_size, model, tile, overlap, device, half, fast, sr_alpha],
            outputs=[out, gallery],
        )

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    demo = build_app()
    demo.launch(server_name=host, server_port=port, share=share)
