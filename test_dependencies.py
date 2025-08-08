#!/usr/bin/env python3
"""
🔧 PixelPure Dependencies Test
=============================
Script kiểm tra và sửa dependencies cho build .exe

Tác giả: TNI Tech Solutions  
Ngày tạo: August 2025
"""

import sys
import subprocess
import importlib

def test_import(package_name, import_name=None):
    """Test import một package"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: {e}")
        return False

def install_package(package_name):
    """Cài đặt package"""
    print(f"🔧 Installing {package_name}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                      check=True, capture_output=True)
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def main():
    """Main function"""
    print("🔧 PixelPure Dependencies Test")
    print("=" * 40)
    
    # Test core dependencies
    dependencies = [
        ("PySide6", "PySide6"),
        ("PyInstaller", "PyInstaller"),
        ("Torch", "torch"),
        ("TorchVision", "torchvision"),
        ("OpenCLIP", "open_clip"),
        ("Pillow", "PIL"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("ImageHash", "imagehash"),
        ("Send2Trash", "send2trash"),
    ]
    
    print("\n📋 Checking dependencies...")
    failed = []
    
    for package_name, import_name in dependencies:
        if not test_import(package_name, import_name):
            failed.append((package_name, import_name))
    
    # Install missing packages
    if failed:
        print(f"\n🔧 Installing {len(failed)} missing packages...")
        
        install_map = {
            "PySide6": "PySide6",
            "PyInstaller": "pyinstaller", 
            "Torch": "torch",
            "TorchVision": "torchvision",
            "OpenCLIP": "open-clip-torch",
            "Pillow": "Pillow",
            "OpenCV": "opencv-python",
            "NumPy": "numpy",
            "ImageHash": "imagehash",
            "Send2Trash": "send2trash",
        }
        
        for package_name, import_name in failed:
            pip_name = install_map.get(package_name, package_name.lower())
            install_package(pip_name)
        
        print("\n🔄 Re-testing after installation...")
        for package_name, import_name in failed:
            test_import(package_name, import_name)
    
    # Test PySide6 specifically
    print("\n🎯 Testing PySide6 components...")
    pyside6_components = [
        "PySide6.QtCore",
        "PySide6.QtGui", 
        "PySide6.QtWidgets",
        "shiboken6"
    ]
    
    for component in pyside6_components:
        test_import(component, component)
    
    # Test PyTorch components  
    print("\n🤖 Testing PyTorch components...")
    torch_components = [
        "torch.nn",
        "torch.nn.functional", 
        "torchvision.transforms"
    ]
    
    for component in torch_components:
        test_import(component, component)
    
    print("\n✅ Dependency check completed!")
    print("\n🚀 You can now run: python build_exe.py")

if __name__ == "__main__":
    main()
