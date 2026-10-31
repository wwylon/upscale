# -*- mode: python ; coding: utf-8 -*-
# Build from repo root:
#   py -3 -m PyInstaller desktop/upscale-cpu.spec --noconfirm
#
# Produces dist/Upscale/ (onedir). Then compile desktop/installer.iss with Inno Setup
# to get Upscale-Setup-cpu.exe — one file to download and install.
#
# True --onefile with Torch is possible but multi-GB and very slow to start;
# the Setup.exe is the supported “download one file” product.

from pathlib import Path

block_cipher = None
# SPECPATH is the directory containing this .spec file (= desktop/)
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
        "upscaler.engine",
        "upscaler.cli",
        "upscaler.mcp_server",
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
        "pydantic",
        "PIL",
        "numpy",
        "torch",
        "spandrel",
        "tifffile",
        "click",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(desktop / "runtime_hook.py")],
    excludes=[
        "tkinter",
        "matplotlib",
        "gradio",
        "pandas",
        "scipy",
        "IPython",
        "notebook",
        "jupyter",
        "transformers",
        "tensorflow",
        "tensorboard",
        "onnxruntime",
        "onnx",
        "pyarrow",
        "sqlalchemy",
        "lxml",
        "triton",
        "torchvision",
        "torchaudio",
        "cv2",
        "sklearn",
        "pytest",
        "black",
        "mypy",
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
