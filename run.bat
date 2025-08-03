@echo off
echo Starting PixelPure Application...
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Virtual environment activated.
echo.
echo Running PixelPure...
python main.py
echo.
echo Application closed.
pause
