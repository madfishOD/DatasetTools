@echo off
setlocal

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"

REM Construct absolute paths for the venv and plugins directory
set "VENV_DIR=%SCRIPT_DIR%venv"
set "PLUGINS_DIR=%SCRIPT_DIR%venv\fiftyone_plugins"

REM Set the environment variable so FiftyOne knows where to find the plugins
set FIFTYONE_PLUGINS_DIR=%PLUGINS_DIR%

echo.
echo =================================================
echo             FiftyOne Launch Helper
echo =================================================
echo.
echo Activating virtual environment from:
echo %VENV_DIR%
echo.
call "%VENV_DIR%\Scripts\activate.bat"

echo Setting plugins directory to:
echo %FIFTYONE_PLUGINS_DIR%
echo.

echo Launching FiftyOne App...
echo (Close the web browser and this window to exit)
echo.
fiftyone app launch

REM Deactivate the virtual environment when the app is closed
deactivate

endlocal
pause