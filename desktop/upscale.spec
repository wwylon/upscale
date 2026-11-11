# PyInstaller stub — build from repo root after pip install -e ".[desktop,server]"
#
#   pyinstaller desktop/upscale.spec
#
# Expected data files: ui/, locales/, docs/help/, upscale package.
# Torch + models are large; ship models separately or download on first run.

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['desktop/app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('ui', 'ui'),
        ('locales', 'locales'),
        ('docs/help', 'docs/help'),
    ],
    hiddenimports=['upscale.server', 'uvicorn', 'webview'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='Upscale',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Upscale',
)
