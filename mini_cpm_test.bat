@echo off
setlocal

REM --- Download and register MiniCPM-V-4_5 as zoo model---
echo.
echo Download and register MiniCPM-V-4_5 as zoo model...
venv\Scripts\python.exe mini_cpm_test.py

endlocal
pause