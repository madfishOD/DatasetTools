@echo off
setlocal
chcp 65001 >nul
set "HF_HOME=%~dp0runtime\hf-cache"
set "PYTHONNOUSERSITE=1"
set "AUTO_PY=%~dp0runtime\python\python.exe"
if defined AUTO_CAPTIONING_PYTHON (
  set "AUTO_PY=%AUTO_CAPTIONING_PYTHON%"
  goto run
)
if not exist "%~dp0runtime\ready.txt" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_portable.ps1"
  if errorlevel 1 goto setup_failed
)
:run
if "%~1"=="" (
  "%AUTO_PY%" -u "%~dp0auto_captioning_tool.py" --interactive
) else (
  "%AUTO_PY%" -u "%~dp0auto_captioning_tool.py" %*
)
set "AUTO_EXIT=%ERRORLEVEL%"
echo Exit code: %AUTO_EXIT%
if not defined AUTO_CAPTIONING_NO_PAUSE pause
exit /b %AUTO_EXIT%
:setup_failed
echo Setup failed. See the error above; rerun this BAT to retry.
if not defined AUTO_CAPTIONING_NO_PAUSE pause
exit /b 1
