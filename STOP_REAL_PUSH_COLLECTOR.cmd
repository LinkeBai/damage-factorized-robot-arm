@echo off
setlocal
title Stop Real Push Collector
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
  "$items = Get-CimInstance Win32_Process ^| Where-Object { $_.Name -match '^python(.exe)?$' -and ($_.CommandLine -match 'collect_real_push_automated\.py' -or $_.CommandLine -match 'serve_dual_camera_preview\.py') }; if(-not $items){Write-Host 'Collector is not running.'} else {$items ^| ForEach-Object {Write-Host ('Stopping PID '+$_.ProcessId); Stop-Process -Id $_.ProcessId -Force}}"
echo.
echo Collector and camera preview stopped.
pause
endlocal
