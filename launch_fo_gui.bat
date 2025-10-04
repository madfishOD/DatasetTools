@echo off
setlocal

echo.
echo =================================================
echo             FiftyOne Launch Helper
echo =================================================
echo.

echo Launching FiftyOne App...
echo (Close the web browser and this window to exit)
venv\Scripts\python.exe launch_fo_gui.py

REM Deactivate the virtual environment when the app is closed
deactivate

endlocal
pause