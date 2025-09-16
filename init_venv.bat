@echo off
setlocal ENABLEEXTENSIONS

rem ---------- Config ----------
rem Default: create ".venv" next to this .bat. You can pass a custom dir as %1
set "VENV_DIR=%~dp0.venv"
if not "%~1"=="" set "VENV_DIR=%~1"

echo.
echo [*] Creating virtual environment at: "%VENV_DIR%"
echo.

rem ---------- Prefer the Python launcher (py) ----------
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    call :mkvenv_py "%VENV_DIR%"
    if %ERRORLEVEL% NEQ 0 goto try_python_exe
) else (
    goto try_python_exe
)
goto activate

:try_python_exe
where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python -m venv "%VENV_DIR%" || goto venv_error
) else (
    echo [!] Python not found on PATH. Install Python and/or enable the "py" launcher.
    goto end
)

:activate
echo [*] Activating venv (cmd)...
call "%VENV_DIR%\Scripts\activate.bat" || goto activate_error

echo [*] Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel

if exist "%~dp0requirements.txt" (
    echo [*] Installing requirements.txt next to this script...
    pip install -r "%~dp0requirements.txt"
)

echo.
echo [OK] Venv ready.
echo     - cmd.exe     : call "%VENV_DIR%\Scripts\activate.bat"
echo     - PowerShell  : %VENV_DIR:\=\\%\Scripts\\Activate.ps1
echo     - Deactivate  : deactivate
echo.
pause
goto end

:mkvenv_py
rem Try specific versions first (3.11 preferred for ML), then fallback
py -3.11 -m venv "%~1" 2>nul && exit /b 0
py -3.12 -m venv "%~1" 2>nul && exit /b 0
py -3     -m venv "%~1"        && exit /b 0
exit /b 1

:venv_error
echo [!] Failed to create venv at "%VENV_DIR%".
pause
goto end

:activate_error
echo [!] Venv created, but activation failed. Activate later with:
echo     call "%VENV_DIR%\Scripts\activate.bat"
pause

:end
endlocal
