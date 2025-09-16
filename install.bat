@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ===================== init_venv_diag.bat =====================
REM Forces the console to stay open even on errors and logs next to this BAT.

set "HERE=%~dp0"
set "LOG=%HERE%init_venv.log"
echo === START === > "%LOG%"
echo [*] Log: "%LOG%"
echo [*] Working dir: "%HERE%" >>"%LOG%"
echo [*] Running as: %USERNAME% >>"%LOG%"

set "VENV_DIR=%HERE%.venv"
if not "%~1"=="" set "VENV_DIR=%~1"

echo [*] Target venv: "%VENV_DIR%"
echo [*] Target venv: "%VENV_DIR%" >>"%LOG%"

REM Find Python
set "PYEXE="
where py >>"%LOG%" 2>&1 && set "PYEXE=py"
if "%PYEXE%"=="" where python >>"%LOG%" 2>&1 && set "PYEXE=python"
if "%PYEXE%"=="" (
  echo [!] Python not found on PATH. >>"%LOG%"
  echo [!] Python not found on PATH.
  goto END
)

REM Create venv if missing
if exist "%VENV_DIR%\Scripts\python.exe" (
  echo [=] Venv already exists. >>"%LOG%"
) else (
  echo [*] Creating venv... >>"%LOG%"
  "%PYEXE%" -m venv "%VENV_DIR%" >>"%LOG%" 2>&1
)

REM Activate
call "%VENV_DIR%\Scripts\activate.bat" >>"%LOG%" 2>&1

REM Show python info
python -V >>"%LOG%" 2>&1
where python >>"%LOG%" 2>&1

REM Upgrade tools
echo [*] Upgrading pip... >>"%LOG%"
python -m pip install --upgrade pip setuptools wheel >>"%LOG%" 2>&1

REM Install requests if missing
python -m pip show requests >>"%LOG%" 2>&1 || python -m pip install requests >>"%LOG%" 2>&1

echo [✓] Done. See the log for details.
echo.
echo === LOG TAIL ===
type "%LOG%"
echo === END LOG ===
echo.
pause
:END
endlocal
