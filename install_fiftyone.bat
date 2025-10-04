@echo off
setlocal

REM --- Create and install FiftyOne and dependencies ---
echo.
echo Installing FiftyOne and dependencies...
python install_fiftyone.py

REM --- Set the FIFTYONE_PLUGINS_DIR environment variable persistently ---
echo.
echo Setting persistent environment variable for plugins...
set "SCRIPT_DIR=%~dp0"
REM This path now correctly points to the plugins directory inside the venv folder.
set "PLUGINS_DIR=%SCRIPT_DIR%\fiftyone_plugins"

REM --- Set the FIFTYONE_DATABASE_DIR environment variable persistently ---
echo.
echo Setting persistent environment variable for database...
REM This path now points to the fiftyone db directory the script root folder.
set "DATABASE_DIR=%SCRIPT_DIR%\fiftyone_db"

REM --- Set the FIFTYONE_MODEL_ZOO_DIR environment variable persistently ---
echo.
echo Setting persistent environment variable for database...
REM This path now points to the fiftyone db directory the script root folder.
set "ZOO_MODELS_DIR=%SCRIPT_DIR%\fiftyone_zoo_models"

REM Convert to absolute path
for %%i in ("%PLUGINS_DIR%") do set "ABS_PLUGINS_DIR=%%~fi"
for %%i in ("%DATABASE_DIR%") do set "ABS_DATABASE_DIR=%%~fi"
for %%i in ("%ZOO_MODELS_DIR%") do set "ABS_ZOO_MODELS_DIR=%%~fi"

echo Setting FIFTYONE_PLUGINS_DIR to: %ABS_PLUGINS_DIR%
setx FIFTYONE_PLUGINS_DIR "%ABS_PLUGINS_DIR%"

echo Setting FIFTYONE_DATABASE_DIR to: %ABS_DATABASE_DIR%
setx FIFTYONE_DATABASE_DIR "%ABS_DATABASE_DIR%"

echo Setting FIFTYONE_MODEL_ZOO_DIR to: %ABS_DATABASE_DIR%
setx FIFTYONE_MODEL_ZOO_DIR "%ABS_ZOO_MODELS_DIR%"

REM --- Activate venv and install plugins ---
echo.
echo Installing FiftyOne plugins...
venv\Scripts\python.exe install_plugins.py

echo 

echo.
echo =====================================================================
echo  IMPORTANT: Please close and reopen this terminal window
echo  for the environment variable changes to take effect.
echo =====================================================================
echo.

endlocal
pause