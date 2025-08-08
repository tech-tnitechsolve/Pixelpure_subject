#!/usr/bin/env python3
"""
🔧 PixelPure Windows .exe Builder
==============================
Script tự động để build PixelPure thành file .exe cho Windows

Tác giả: TNI Tech Solutions
Ngày tạo: August 2025
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# ANSI Colors for console output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_step(message):
    """In bước hiện tại với màu"""
    print(f"{Colors.OKCYAN}🔧 {message}{Colors.ENDC}")

def print_success(message):
    """In thông báo thành công"""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

def print_error(message):
    """In thông báo lỗi"""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")

def print_warning(message):
    """In cảnh báo"""
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")

def check_requirements():
    """Kiểm tra yêu cầu hệ thống"""
    print_step("Kiểm tra yêu cầu hệ thống...")
    
    # Kiểm tra Python version
    if sys.version_info < (3, 12):
        print_error(f"Cần Python 3.12+. Hiện tại: {sys.version}")
        return False
    
    # Kiểm tra PyInstaller
    try:
        import PyInstaller
        print_success(f"PyInstaller đã cài đặt: {PyInstaller.__version__}")
    except ImportError:
        print_warning("PyInstaller chưa cài đặt. Đang cài đặt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print_success("PyInstaller đã được cài đặt")
    
    # Kiểm tra PySide6
    try:
        import PySide6
        print_success(f"PySide6 đã cài đặt: {PySide6.__version__}")
    except ImportError:
        print_error("PySide6 chưa cài đặt!")
        print_warning("Đang cài đặt PySide6...")
        subprocess.run([sys.executable, "-m", "pip", "install", "PySide6"], check=True)
        print_success("PySide6 đã được cài đặt")
    
    # Kiểm tra các dependencies quan trọng
    required_packages = [
        'torch', 'torchvision', 'open_clip_torch', 
        'PIL', 'cv2', 'numpy', 'imagehash', 'send2trash'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'open_clip_torch':
                import open_clip
            else:
                __import__(package)
            print_success(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print_warning(f"  ⚠️  {package} - missing")
    
    if missing_packages:
        print_warning(f"Thiếu {len(missing_packages)} packages. Đang cài đặt...")
        for package in missing_packages:
            if package == 'open_clip_torch':
                subprocess.run([sys.executable, "-m", "pip", "install", "open-clip-torch"], check=True)
            elif package == 'PIL':
                subprocess.run([sys.executable, "-m", "pip", "install", "Pillow"], check=True)
            elif package == 'cv2':
                subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python"], check=True)
            else:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        print_success("Tất cả dependencies đã được cài đặt")
    
    return True

def create_build_directory():
    """Tạo thư mục build"""
    print_step("Tạo thư mục build...")
    
    build_dir = Path("build_exe")
    dist_dir = Path("dist")
    
    # Xóa thư mục cũ nếu tồn tại
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    build_dir.mkdir(exist_ok=True)
    print_success("Thư mục build đã được tạo")

def create_spec_file():
    """Tạo file .spec cho PyInstaller"""
    print_step("Tạo file .spec...")
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path
import PySide6

# Đường dẫn project
project_path = Path.cwd()
pyside6_path = Path(PySide6.__file__).parent

a = Analysis(
    ['main.py'],
    pathex=[str(project_path)],
    binaries=[
        # Include PySide6 binaries explicitly
        (str(pyside6_path / "*.dll"), "PySide6"),
        (str(pyside6_path / "*.pyd"), "PySide6"),
    ],
    datas=[
        # Include PySide6 data files
        (str(pyside6_path), "PySide6"),
        # Thêm các file UI
        ('app_ui.py', '.'),
        ('auto_processor.py', '.'),
        ('cache_manager.py', '.'),
        ('improved_ui_components.py', '.'),
        ('result_dashboard.py', '.'),
        ('speed_config.py', '.'),
        ('core', 'core'),
    ],
    hiddenimports=[
        # PySide6 complete imports
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui', 
        'PySide6.QtWidgets',
        'PySide6.QtOpenGL',
        'shiboken6',
        # AI/ML libraries
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torchvision',
        'torchvision.transforms',
        'open_clip',
        'open_clip.model',
        'open_clip.transform',
        # Image processing
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'cv2',
        'numpy',
        'imagehash',
        # System utilities
        'send2trash',
        'pathlib',
        'json',
        'time',
        'threading',
        'queue',
        'typing',
        'dataclasses',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'jupyter',
        'IPython',
        'pandas',
        'scipy',
        'sklearn',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PixelPure',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if Path('icon.ico').exists() else None,
    version_file='version_info.txt' if Path('version_info.txt').exists() else None,
    # Additional options for better compatibility
    uac_admin=False,
    uac_uiaccess=False,
)

# Create COLLECT for better structure (optional)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PixelPure_dist'
)
'''
    
    with open("PixelPure.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)
    
    print_success("File .spec đã được tạo")

def create_version_info():
    """Tạo file version info cho .exe"""
    print_step("Tạo version info...")
    
    version_info = '''# UTF-8
#
# Version information for PixelPure.exe
#
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1,3,0,0),
    prodvers=(1,3,0,0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
    ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'TNI Tech Solutions'),
        StringStruct(u'FileDescription', u'PixelPure - AI-Powered Image Analysis Tool'),
        StringStruct(u'FileVersion', u'1.3.0.0'),
        StringStruct(u'InternalName', u'PixelPure'),
        StringStruct(u'LegalCopyright', u'Copyright © 2025 TNI Tech Solutions'),
        StringStruct(u'OriginalFilename', u'PixelPure.exe'),
        StringStruct(u'ProductName', u'PixelPure'),
        StringStruct(u'ProductVersion', u'1.3.0.0')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
    
    with open("version_info.txt", "w", encoding="utf-8") as f:
        f.write(version_info)
    
    print_success("Version info đã được tạo")

def build_exe():
    """Build file .exe"""
    print_step("Bắt đầu build file .exe...")
    print_warning("Quá trình này có thể mất 10-15 phút...")
    
    # Build với PyInstaller
    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm", 
        "PixelPure.spec"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print_success("Build thành công!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Build thất bại: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def create_installer():
    """Tạo installer script"""
    print_step("Tạo installer script...")
    
    installer_content = '''@echo off
echo.
echo ===============================================
echo    PixelPure v1.3 - Windows Installer
echo    TNI Tech Solutions - August 2025
echo ===============================================
echo.

echo 🔧 Đang cài đặt PixelPure...

REM Tạo thư mục cài đặt
set INSTALL_DIR=C:\\Program Files\\PixelPure
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

REM Copy files
echo 📁 Copying files...
xcopy /E /I /Y "dist\\PixelPure.exe" "%INSTALL_DIR%\\"

REM Tạo shortcut trên Desktop
echo 🔗 Tạo shortcut...
set DESKTOP=%USERPROFILE%\\Desktop
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%DESKTOP%\\PixelPure.lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%INSTALL_DIR%\\PixelPure.exe" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs
cscript CreateShortcut.vbs
del CreateShortcut.vbs

echo.
echo ✅ Cài đặt hoàn tất!
echo 🚀 Bạn có thể chạy PixelPure từ Desktop hoặc:
echo    %INSTALL_DIR%\\PixelPure.exe
echo.
pause
'''
    
    with open("installer.bat", "w", encoding="utf-8") as f:
        f.write(installer_content)
    
    print_success("Installer script đã được tạo")

def create_readme_exe():
    """Tạo README cho bản .exe"""
    print_step("Tạo README cho bản .exe...")
    
    readme_content = '''# 🖼️ PixelPure v1.3 - Windows Executable

## 📦 Hướng dẫn cài đặt

### **Phương pháp 1: Chạy trực tiếp (Khuyến nghị)**
1. Tải file `PixelPure.exe` từ thư mục `dist/`
2. Double-click để chạy ngay lập tức
3. Không cần cài đặt Python hay dependencies

### **Phương pháp 2: Cài đặt vào hệ thống**
1. Chạy `installer.bat` với quyền Administrator
2. PixelPure sẽ được cài vào `C:\\Program Files\\PixelPure\\`
3. Shortcut sẽ được tạo trên Desktop

## 📋 Yêu cầu hệ thống
- **OS:** Windows 10/11 (64-bit)
- **RAM:** 4GB+ (8GB khuyến nghị)
- **Storage:** 2GB+ dung lượng trống
- **Internet:** Cần kết nối cho lần đầu tải AI models

## ⚡ Tính năng
- ✅ **Portable:** Chạy mà không cần cài đặt Python
- ✅ **Self-contained:** Tất cả dependencies đã được đóng gói
- ✅ **AI-Powered:** CLIP model tự động tải về khi cần
- ✅ **User-friendly:** Giao diện đơn giản, dễ sử dụng

## 🚀 Cách sử dụng
1. Chạy `PixelPure.exe`
2. Drag & drop ảnh vào ứng dụng
3. Nhấn "Bắt đầu quét"
4. Xem kết quả và xử lý tự động

## 📞 Hỗ trợ
- **GitHub:** https://github.com/tech-tnitechsolve/Pixelpure_subject
- **Email:** support@tnitechsolve.com

---
**Made with ❤️ by TNI Tech Solutions**
'''
    
    with open("README_EXE.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print_success("README cho bản .exe đã được tạo")

def cleanup():
    """Dọn dẹp files tạm"""
    print_step("Dọn dẹp files tạm...")
    
    temp_files = ["PixelPure.spec", "version_info.txt"]
    temp_dirs = ["build", "__pycache__"]
    
    for file in temp_files:
        if Path(file).exists():
            Path(file).unlink()
    
    for dir in temp_dirs:
        if Path(dir).exists():
            shutil.rmtree(dir)
    
    print_success("Dọn dẹp hoàn tất")

def main():
    """Main function"""
    print(f"{Colors.HEADER}")
    print("🔧 PixelPure Windows .exe Builder")
    print("=" * 40)
    print("TNI Tech Solutions - August 2025")
    print(f"{Colors.ENDC}")
    
    try:
        # Kiểm tra yêu cầu
        if not check_requirements():
            return
        
        # Tạo build directory
        create_build_directory()
        
        # Tạo các file cần thiết
        create_spec_file()
        create_version_info()
        create_installer()
        create_readme_exe()
        
        # Build .exe
        if build_exe():
            print_success("🎉 Build thành công!")
            print(f"{Colors.OKGREEN}")
            print("📁 Files đã được tạo:")
            print("  📄 dist/PixelPure.exe       - Ứng dụng chính")
            print("  📄 installer.bat           - Script cài đặt") 
            print("  📄 README_EXE.md          - Hướng dẫn sử dụng")
            print(f"{Colors.ENDC}")
            
            # Hỏi có muốn dọn dẹp không
            choice = input("\n🔧 Có muốn dọn dẹp files tạm? (y/n): ").lower()
            if choice in ['y', 'yes']:
                cleanup()
        else:
            print_error("Build thất bại!")
            
    except KeyboardInterrupt:
        print_error("\nBuild bị hủy bởi người dùng")
    except Exception as e:
        print_error(f"Lỗi không mong muốn: {e}")

if __name__ == "__main__":
    main()
