@echo off
title IPWM Final Push Day
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_final_real_ipwm_day.ps1"
if errorlevel 1 echo STARTUP FAILED - do not run a formal trial.
pause
