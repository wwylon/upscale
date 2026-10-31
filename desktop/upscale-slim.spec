# -*- mode: python ; coding: utf-8 -*-
# Slim Windows desktop — NO PyTorch. Uses ONNX Runtime.
# From repo root:
#   py -3 -m PyInstaller desktop/upscale-slim.spec --noconfirm --distpath dist --workpath build/pyinstaller-slim
#
# Target: tens of MB (+ ONNX models downloaded separately), not multi-GB.

from pathlib import Path

block_cipher = None
desktop = Path(SPECPATH)
repo = desktop.parent

a = Analysis(
    [str(desktop / "app.py")],
    pathex=[str(repo)],
    binaries=[],
    datas=[
        (str(repo / "ui"), "ui"),
        (str(repo / "locales"), "locales"),
        (str(repo / "docs"), "docs"),
        (str(repo / "upscaler"), "upscaler"),
    ],
    hiddenimports=[
        "upscaler",
        "upscaler.server",
        "upscaler.backends",
        "upscaler.backends.onnx_rt",
        "upscaler.engine",
        "upscaler.cli",
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "fastapi",
        "webview",
        "multipart",
        "python_multipart",
        "pydantic",
        "PIL",
        "numpy",
        "onnxruntime",
        "click",
        "tifffile",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(desktop / "runtime_hook.py")],
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "spandrel",
        "triton",
        "tensorflow",
        "tensorboard",
        "transformers",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "gradio",
        "IPython",
        "notebook",
        "cv2",
        "onnx",  # converter only; runtime is onnxruntime
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Upscale",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(desktop / "upscale.ico") if (desktop / "upscale.ico").is_file() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Upscale",
)
