@echo off
echo.
echo ===============================================
echo    PixelPure v1.3 - Windows Installer
echo    TNI Tech Solutions - August 2025
echo ===============================================
echo.

echo 🔧 Đang cài đặt PixelPure...

REM Tạo thư mục cài đặt
set INSTALL_DIR=C:\Program Files\PixelPure
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

REM Copy files
echo 📁 Copying files...
xcopy /E /I /Y "dist\PixelPure.exe" "%INSTALL_DIR%\"

REM Tạo shortcut trên Desktop
echo 🔗 Tạo shortcut...
set DESKTOP=%USERPROFILE%\Desktop
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%DESKTOP%\PixelPure.lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%INSTALL_DIR%\PixelPure.exe" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs
cscript CreateShortcut.vbs
del CreateShortcut.vbs

echo.
echo ✅ Cài đặt hoàn tất!
echo 🚀 Bạn có thể chạy PixelPure từ Desktop hoặc:
echo    %INSTALL_DIR%\PixelPure.exe
echo.
pause
