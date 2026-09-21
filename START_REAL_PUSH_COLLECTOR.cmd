@echo off
setlocal
title Real Push Collector
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_real_push_collector.ps1"
endlocal
