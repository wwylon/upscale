# Build the *shippable* Windows app (no PyTorch — ONNX Runtime only).
#   .\desktop\build_windows_slim.ps1
#
# Output:
#   dist\Upscale\Upscale.exe
#   desktop\Output\Upscale-Setup-slim.exe   (if Inno Setup installed)
#   desktop\Output\Upscale-portable-slim.zip
#
# Prerequisite (once, on a Torch machine): export at least one ONNX model:
#   py -3 scripts/export_onnx.py 4x-spanx4-ch48 -o models/4x-spanx4-ch48.onnx
# Models are NOT bundled; users download/export into %LOCALAPPDATA%\Upscale\models.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== Sync design UI ==" -ForegroundColor Cyan
py -3 scripts\sync_design_ui.py

Write-Host "== Slim deps ==" -ForegroundColor Cyan
py -3 -m pip install "onnxruntime>=1.17" "fastapi>=0.110" "uvicorn[standard]" "python-multipart" "pywebview>=5.0" pyinstaller -q
py -3 -c "import os; os.environ['UPSCALE_BACKEND']='onnx'; import onnxruntime, fastapi, webview; from upscale.backends import resolve_backend; assert resolve_backend()=='onnx'; print('slim imports ok', onnxruntime.__version__)"

Write-Host "== PyInstaller slim onedir ==" -ForegroundColor Cyan
if (Test-Path dist\Upscale) { Remove-Item -Recurse -Force dist\Upscale }
py -3 -m PyInstaller desktop\upscale-slim.spec --noconfirm --distpath dist --workpath build\pyinstaller-slim
if (-not (Test-Path "dist\Upscale\Upscale.exe")) {
    throw "PyInstaller did not produce dist\Upscale\Upscale.exe"
}

# Fail the build if Torch leaked into the bundle
$torchLeak = Get-ChildItem dist\Upscale -Recurse -Filter "torch*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($torchLeak) {
    throw "Torch DLL found in slim bundle: $($torchLeak.FullName) — slim build is contaminated"
}

$sizeMB = [math]::Round(((Get-ChildItem dist\Upscale -Recurse -File | Measure-Object Length -Sum).Sum) / 1MB, 1)
Write-Host "Slim onedir size: $sizeMB MB (target << 500 MB; ideally < 150 MB)" -ForegroundColor Green
if ($sizeMB -gt 400) {
    Write-Host "WARNING: slim bundle is larger than expected — check excludes." -ForegroundColor Yellow
}

$iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

New-Item -ItemType Directory -Force -Path "desktop\Output" | Out-Null

# Temporary iss tweak: output name Upscale-Setup-slim
$iss = Get-Content desktop\installer.iss -Raw
$issSlim = $iss -replace "OutputBaseFilename=Upscale-Setup-cpu", "OutputBaseFilename=Upscale-Setup-slim"
$issSlim = $issSlim -replace "Compression=lzma2/ultra64", "Compression=lzma"
$issSlim = $issSlim -replace "SolidCompression=yes", "SolidCompression=no"
$tmpIss = "desktop\_installer_slim.iss"
Set-Content -Path $tmpIss -Value $issSlim -Encoding UTF8

if ($iscc) {
    Write-Host "== Inno Setup -> Upscale-Setup-slim.exe ==" -ForegroundColor Cyan
    & $iscc $tmpIss
} else {
    Write-Host "Inno Setup not found — skipping Setup.exe" -ForegroundColor Yellow
}

$seven = @(
    "${env:ProgramFiles}\7-Zip\7z.exe",
    "${env:ProgramFiles(x86)}\7-Zip\7z.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
$zip = "desktop\Output\Upscale-portable-slim.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Push-Location dist\Upscale
try {
    if ($seven) {
        & $seven a -tzip -mx=5 -mmt=on "..\..\desktop\Output\Upscale-portable-slim.zip" "*"
    } else {
        tar -a -cf "..\..\desktop\Output\Upscale-portable-slim.zip" *
    }
} finally {
    Pop-Location
}

Remove-Item $tmpIss -Force -ErrorAction SilentlyContinue

if (Test-Path "desktop\Output\Upscale-Setup-slim.exe") {
    $s = [math]::Round((Get-Item "desktop\Output\Upscale-Setup-slim.exe").Length / 1MB, 1)
    Write-Host "DONE Setup: desktop\Output\Upscale-Setup-slim.exe ($s MB)" -ForegroundColor Green
}
if (Test-Path $zip) {
    $s = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Write-Host "DONE portable: $zip ($s MB)" -ForegroundColor Green
}
Write-Host "Users still need an ONNX model in %LOCALAPPDATA%\Upscale\models (e.g. 4x-spanx4-ch48.onnx)."
