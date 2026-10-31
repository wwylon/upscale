# PyInstaller stub for macOS — build on a Mac with pywebview + upscaler[desktop,server]
#
#   pyinstaller desktop/upscale-macos.spec
#
# Codesign and notarize before distribution; torch wheels are platform-specific.

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
    hiddenimports=['upscaler.server', 'uvicorn', 'webview'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='Upscale',
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
