@echo off
setlocal ENABLEEXTENSIONS
set "HERE=%~dp0"
set "VENV_DIR=%HERE%.venv"

if exist "%VENV_DIR%\Scripts\python.exe" (
  echo [*] Activating venv: "%VENV_DIR%"
  call "%VENV_DIR%\Scripts\activate.bat"
) else (
  echo [*] No venv found at "%VENV_DIR%". Using system Python on PATH.
)

rem Optional: pin a DB directory (uncomment and edit)
rem set "FIFTYONE_DATABASE_DIR=D:\StableDiffusion\DatasetTools\FiftyOne\.fo_db"

echo.
echo [*] Launching Tk GUI...
echo     You can pass args, e.g.:  --db_dir D:\fo_db  --address 127.0.0.1  --caption_ext .txt
echo.
python "%HERE%export_fiftyone_view.py" %*
echo.
pause
endlocal
