@echo off
setlocal EnableExtensions
title PixelPure - Launcher

rem Change working directory to the folder where this script resides
cd /d "%~dp0"

rem Try to activate a virtual environment if present (.venv or venv)
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
  call "venv\Scripts\activate.bat"
) else (
  rem No activate script found; prefer bundled python.exe if present
  if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
  ) else if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
  ) else (
    set "PYTHON=python"
  )
)

rem Run the application (forward any args passed to the .bat)
if defined PYTHON (
  "%PYTHON%" main.py %*
) else (
  python main.py %*
)

set "EC=%ERRORLEVEL%"

rem On error, show message and wait; on success, exit immediately
if not "%EC%"=="0" (
  echo.
  echo Application exited with error code %EC%.
  echo Press any key to close this window.
  pause >nul
)

exit /b %EC%
