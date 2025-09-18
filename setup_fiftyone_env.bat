@echo off
setlocal ENABLEEXTENSIONS

rem =========================
rem Config
rem =========================
set "VENV_DIR=%~dp0.venv"
if not "%~1"=="" set "VENV_DIR=%~1"

echo.
echo [*] Target virtual environment: "%VENV_DIR%"
echo.

rem =========================
rem If a venv already exists, ask to reuse or recreate
rem =========================
if exist "%VENV_DIR%\Scripts\python.exe" (
  echo [!] Found existing venv at "%VENV_DIR%"
  call :ask_yn "Reuse it (keep installed packages) (y/n)? " && (
    echo [*] Activating existing venv...
    call "%VENV_DIR%\Scripts\activate.bat" || goto :activate_error
    goto :install_packages
  )
  echo [*] Removing existing venv...
  rmdir /s /q "%VENV_DIR%" || goto :venv_error
)

rem =========================
rem Create a new venv
rem =========================
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  call :mkvenv_py "%VENV_DIR%"
  if %ERRORLEVEL% NEQ 0 goto :try_python_exe
) else (
  goto :try_python_exe
)
goto :activate_new

:try_python_exe
where python >nul 2>nul
if %ERRORLEVEL%==0 (
  echo [*] Using "python -m venv"...
  python -m venv "%VENV_DIR%" || goto :venv_error
) else (
  echo [!] Python not found on PATH. Install Python 3.11/3.12 (with the "py" launcher) and retry.
  goto :end
)

:activate_new
echo [*] Activating venv (cmd)...
call "%VENV_DIR%\Scripts\activate.bat" || goto :activate_error

rem =========================
rem Install / upgrade core tooling + FiftyOne
rem =========================
:install_packages
echo [*] Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel

echo [*] Installing FiftyOne...
pip install fiftyone || goto :pkg_error
echo [OK] FiftyOne installed.
fiftyone --version
echo.

rem =========================
rem Optional: fiftyone-brain
rem =========================
call :ask_yn "Install fiftyone-brain (y/n)? " && (
  echo [*] Installing fiftyone-brain...
  pip install fiftyone-brain || echo [!] fiftyone-brain install failed.
)
echo.

rem =========================
rem Optional: PyTorch (CPU / CUDA / Skip)
rem =========================
echo Install PyTorch (required by many Brain ops and Florence2/captioning plugins)
set "TORCH_CHOICE="
set /p TORCH_CHOICE="Choose: [C]PU / [G]PU (CUDA 12.1) / [S]kip  > "
if /I "%TORCH_CHOICE%"=="C" (
  echo [*] Installing PyTorch CPU build...
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else if /I "%TORCH_CHOICE%"=="G" (
  echo [*] Installing PyTorch CUDA 12.1 build...
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else (
  echo [*] Skipping PyTorch install.
  set "TORCH_SKIPPED=1"
)
echo.

rem Quick import check (purely informational)
echo [*] Verifying torch import (this may print an error if not installed)...
python -c "import importlib,sys;print('torch:',bool(importlib.util.find_spec('torch')))"
echo.

rem =========================
rem Optional: umap-learn (Embeddings/UMAP)
rem =========================
call :ask_yn "Install umap-learn for Embeddings panel (y/n)? " && (
  echo [*] Installing umap-learn...
  pip install umap-learn || echo [!] umap-learn install failed.
)
echo.

rem =========================
rem Ensure git for plugin downloads
rem =========================
set "GIT_OK="
call :ensure_git

rem =========================
rem Plugins (prompted)
rem =========================
if defined GIT_OK (
  call :ask_yn "Install plugin: voxel51 / plugins / io (y/n)? " && (
    echo [*] Downloading plugin: io ...
    fiftyone plugins download https://github.com/voxel51/fiftyone-plugins/tree/main/plugins/io
  )
  echo.

  call :ask_yn "Install plugin: jacobmarks / clustering-plugin (y/n)? " && (
    echo [*] Downloading plugin: clustering-plugin ...
    fiftyone plugins download https://github.com/jacobmarks/clustering-plugin
  )
  echo.

  call :ask_yn "Install plugin: jacobmarks / fiftyone_florence2_plugin (y/n)? " && (
    echo [*] Downloading plugin: florence2 ...
    fiftyone plugins download https://github.com/jacobmarks/fiftyone_florence2_plugin
  )
  echo.

  call :ask_yn "Install plugin: jacobmarks / fiftyone-image-captioning-plugin (y/n)? " && (
    echo [*] Downloading plugin: image-captioning ...
    fiftyone plugins download https://github.com/jacobmarks/fiftyone-image-captioning-plugin
  )
) else (
  echo [!] "git" not found. Skipping plugin downloads. Install Git for Windows and re-run:
  echo     https://git-scm.com/download/win
)
echo.

rem If Torch was skipped, offer to disable florence2 (avoids error popups)
if defined TORCH_SKIPPED (
  call :ask_yn "Disable @jacobmarks/florence2 plugin (no torch) (y/n)? " && (
    fiftyone plugins disable @jacobmarks/florence2
  )
)

echo.
echo [*] Installed/available plugins:
fiftyone plugins list

echo.
echo [OK] Environment ready.
echo     Activate (cmd):   call "%VENV_DIR%\Scripts\activate.bat"
echo     Activate (PS):    %VENV_DIR:\=\\%\Scripts\\Activate.ps1
echo     Deactivate:       deactivate
echo.
echo     In the App, open the Operator Browser (press the backtick ` ):
echo       - Brain: compute similarity / uniqueness / visualization
echo.
pause
goto :end

rem =========================
rem Functions
rem =========================
:mkvenv_py
rem Prefer 3.11 for ML compat; then 3.12; then "any 3"
echo [*] Using "py" launcher to create venv...
py -3.11 -m venv "%~1" 2>nul && exit /b 0
py -3.12 -m venv "%~1" 2>nul && exit /b 0
py -3     -m venv "%~1"        && exit /b 0
exit /b 1

:ask_yn
setlocal
set "ans="
set /p ans=%~1
if /I "%ans%"=="y"  ( endlocal & exit /b 0 )
if /I "%ans%"=="yes" ( endlocal & exit /b 0 )
endlocal & exit /b 1

:ensure_git
where git >nul 2>nul
if %ERRORLEVEL%==0 (
  set "GIT_OK=1"
  echo [*] Git found.
) else (
  set "GIT_OK="
)
exit /b 0

rem =========================
rem Errors
rem =========================
:pkg_error
echo [!] A pip installation step failed. Check the error above.
pause
goto :end

:venv_error
echo [!] Failed to create venv at "%VENV_DIR%".
pause
goto :end

:activate_error
echo [!] Venv created, but activation failed. Activate manually:
echo     call "%VENV_DIR%\Scripts\activate.bat"
pause
goto :end

:end
endlocal
