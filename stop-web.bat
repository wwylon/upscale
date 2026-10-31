@echo off
setlocal
set "PORT=8787"
echo Stopping Upscale web on port %PORT% ...

powershell -NoProfile -Command ^
  "$conns = Get-NetTCPConnection -LocalPort %PORT% -ErrorAction SilentlyContinue;" ^
  "if (-not $conns) { Write-Host 'Nothing listening on %PORT%.'; exit 0 }" ^
  "$procIds = $conns | Select-Object -ExpandProperty OwningProcess -Unique;" ^
  "foreach ($procId in $procIds) {" ^
  "  try {" ^
  "    $p = Get-CimInstance Win32_Process -Filter (\"ProcessId=$procId\");" ^
  "    Write-Host (\"Killing PID $procId : \" + $p.CommandLine);" ^
  "    Stop-Process -Id $procId -Force -ErrorAction Stop" ^
  "  } catch { Write-Host (\"Could not kill ${procId}: $_\") }" ^
  "}"

echo Done.
pause
