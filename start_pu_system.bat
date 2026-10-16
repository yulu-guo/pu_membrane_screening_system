@echo off
cd /d "%~dp0"

set "PYTHON=%USERPROFILE%\anaconda3\envs\xiao\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo Starting PU membrane property system...
echo Keep this window open while using the website.
echo Website: http://127.0.0.1:8766
echo.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8766'"
"%PYTHON%" app.py --host 127.0.0.1 --port 8766

echo.
echo The server has stopped. Press any key to close this window.
pause >nul
