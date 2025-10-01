@echo off
setlocal

REM --- Create and install FiftyOne and dependencies ---
echo.
echo Installing FiftyOne and dependencies...
python install_fiftyone.py

REM --- Activate venv and install plugins ---
echo.
echo Installing FiftyOne plugins...
venv\Scripts\python.exe install_plugins.py

REM --- Set the FIFTYONE_PLUGINS_DIR environment variable persistently ---
echo.
echo Setting persistent environment variable for plugins...
set "SCRIPT_DIR=%~dp0"
REM This path now correctly points to the plugins directory inside the venv folder.
set "PLUGINS_DIR=%SCRIPT_DIR%venv\fiftyone_plugins"

REM Convert to absolute path
for %%i in ("%PLUGINS_DIR%") do set "ABS_PLUGINS_DIR=%%~fi"

echo Setting FIFTYONE_PLUGINS_DIR to: %ABS_PLUGINS_DIR%
setx FIFTYONE_PLUGINS_DIR "%ABS_PLUGINS_DIR%"

echo.
echo =====================================================================
echo  IMPORTANT: Please close and reopen this terminal window
echo  for the environment variable changes to take effect.
echo =====================================================================
echo.

endlocal
pause