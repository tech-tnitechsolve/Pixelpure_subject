# 🔧 PixelPure PowerShell Build Script
# ====================================
# Script PowerShell để build PixelPure thành .exe

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   PixelPure Windows .exe PowerShell Builder" -ForegroundColor Yellow
Write-Host "   TNI Tech Solutions - August 2025" -ForegroundColor Green  
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra Python
Write-Host "🔧 Kiểm tra Python..." -ForegroundColor Blue
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python không được tìm thấy!" -ForegroundColor Red
    Write-Host "📋 Vui lòng cài đặt Python 3.12+ từ: https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "🔧 Cài đặt PyInstaller (nếu chưa có)..." -ForegroundColor Blue
pip install pyinstaller

Write-Host ""
Write-Host "🚀 Bắt đầu build process..." -ForegroundColor Green
python build_exe.py

Write-Host ""
Write-Host "✅ Build process hoàn tất!" -ForegroundColor Green
Write-Host "📁 Kiểm tra thư mục dist/ để tìm PixelPure.exe" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"
