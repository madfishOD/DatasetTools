@echo off
setlocal
chcp 65001 >nul
set "PYTHONNOUSERSITE=1"
set "AUTO_PY=%~dp0runtime\python\python.exe"
if defined AUTO_CAPTIONING_PYTHON set "AUTO_PY=%AUTO_CAPTIONING_PYTHON%"
if not defined AUTO_CAPTIONING_PYTHON if not exist "%~dp0runtime\ready.txt" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_portable.ps1"
  if errorlevel 1 exit /b 1
)
"%AUTO_PY%" -m pip install --only-binary=:all: -r "%~dp0requirements-gui.txt"
if errorlevel 1 exit /b 1
"%AUTO_PY%" "%~dp0gui.py" %*
exit /b %ERRORLEVEL%
