@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Upscale Web

set "PORT=8787"
set "URL=http://127.0.0.1:%PORT%/en/"

rem Prefer a real Windows python.exe (Ctrl+C reaches uvicorn more reliably than py).
rem Skip PATH entries that cannot import numpy (common: MSYS python without pip/deps).
set "PY_LAUNCH="
for /f "delims=" %%P in ('where python 2^>nul') do (
  "%%P" -c "import numpy" >nul 2>&1
  if not errorlevel 1 (
    set "PY_LAUNCH=%%P"
    goto :have_py
  )
)
where py >nul 2>&1
if not errorlevel 1 (
  py -3 -c "import numpy" >nul 2>&1
  if not errorlevel 1 set "PY_LAUNCH=py -3"
)
:have_py
if not defined PY_LAUNCH (
  echo No usable Python 3 with numpy found.
  echo Install deps with:  py -3 -m pip install -e ".[server]"
  echo Or fix PATH so Windows Python comes before MSYS.
  pause
  exit /b 1
)

rem Free the port if a previous serve is stuck (Ctrl+C often fails on Windows).
powershell -NoProfile -Command ^
  "$conns = Get-NetTCPConnection -LocalPort %PORT% -ErrorAction SilentlyContinue;" ^
  "foreach ($c in $conns) { try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue } catch {} }" >nul 2>&1

echo Starting Upscale web UI at %URL%
echo Waiting until the server is ready, then opening the browser...
echo Close this window or run stop-web.bat to stop the server.
echo.

rem Start serve in the background of this console, wait for HTTP 200, then open the browser.
rem Keeps server logs in this window; Ctrl+C / closing the window stops the server.
start /b "" %PY_LAUNCH% -m upscaler serve --host 127.0.0.1 --port %PORT%

powershell -NoProfile -Command ^
  "$url = '%URL%';" ^
  "$deadline = (Get-Date).AddSeconds(60);" ^
  "do {" ^
  "  try {" ^
  "    $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1;" ^
  "    if ($r.StatusCode -eq 200) { exit 0 }" ^
  "  } catch {}" ^
  "  Start-Sleep -Milliseconds 250" ^
  "} while ((Get-Date) -lt $deadline);" ^
  "Write-Host 'Server did not become ready within 60s.'; exit 1"
if errorlevel 1 (
  echo.
  echo Failed to start. Install server extras:
  echo   %PY_LAUNCH% -m pip install -e ".[server]"
  echo Or run stop-web.bat and try again.
  pause
  exit /b 1
)

start "" "%URL%"

rem Keep this window open while the server is listening (start /b already exited).
:wait_loop
timeout /t 2 /nobreak >nul
powershell -NoProfile -Command ^
  "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 goto wait_loop

echo.
echo Server stopped.
exit /b 0
