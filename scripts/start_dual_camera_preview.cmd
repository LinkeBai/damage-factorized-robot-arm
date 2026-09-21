@echo off
setlocal
cd /d "%~dp0.."
:restart
".venv-cuda\Scripts\python.exe" scripts\serve_dual_camera_preview.py
echo Camera preview stopped. Restarting in 2 seconds...
timeout /t 2 /nobreak >nul
goto restart
