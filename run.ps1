# PixelPure Application Launcher
Write-Host "🎨 Starting PixelPure Application..." -ForegroundColor Cyan
Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

Write-Host "✅ Virtual environment activated." -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Running PixelPure..." -ForegroundColor Cyan

# Run the application
python main.py

Write-Host ""
Write-Host "👋 Application closed." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
