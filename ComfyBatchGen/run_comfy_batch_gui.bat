@echo off
setlocal

rem Folder containing this BAT (and comfy_batch_gui.py)
set "SCRIPT_DIR=%~dp0"

rem Resolve the parent folder (..), then append .venv
for %%I in ("%SCRIPT_DIR%..") do set "VENV_DIR=%%~fI\.venv"
set "PYEXE=%VENV_DIR%\Scripts\python.exe"
set "APP=%SCRIPT_DIR%comfy_batch_gui.py"

echo [info] Script dir : %SCRIPT_DIR%
echo [info] Venv dir   : %VENV_DIR%
echo [info] Python exe : %PYEXE%
echo [info] App        : %APP%

if not exist "%PYEXE%" (
  echo [error] Expected venv Python not found at:
  echo         "%PYEXE%"
  echo Ensure the venv exists at "..\.venv" relative to this folder.
  pause
  exit /b 1
)

if not exist "%APP%" (
  echo [error] Could not find comfy_batch_gui.py at:
  echo         "%APP%"
  pause
  exit /b 1
)

echo [run] Launching GUI...
"%PYEXE%" "%APP%"
exit /b %ERRORLEVEL%
