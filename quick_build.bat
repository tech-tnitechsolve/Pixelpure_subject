@echo off
REM 🚀 PixelPure Quick Build Script
REM ===============================
REM Script nhanh để build PixelPure thành .exe

echo.
echo ===============================================
echo    PixelPure Windows .exe Quick Builder
echo    TNI Tech Solutions - August 2025  
echo ===============================================
echo.

echo 🔧 Kiểm tra Python...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python không được tìm thấy!
    echo 📋 Vui lòng cài đặt Python 3.12+ từ: https://python.org
    pause
    exit /b 1
)

echo.
echo 🔧 Cài đặt PyInstaller (nếu chưa có)...
pip install pyinstaller

echo.
echo 🚀 Bắt đầu build process...
python build_exe.py

echo.
echo ✅ Build process hoàn tất!
echo 📁 Kiểm tra thư mục dist/ để tìm PixelPure.exe
echo.
pause
