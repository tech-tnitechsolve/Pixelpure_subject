# install_requirements.py
#
# Phiên bản V26: Cài đặt độc lập và thông minh.
# - [INDEPENDENT] Chứa danh sách đầy đủ các thư viện, không phụ thuộc vào file khác.
# - [SMART] Vẫn giữ lại tính năng tự động phát hiện GPU để cài đặt phiên bản PyTorch tối ưu nhất.

import subprocess
import sys
import os

# --- Cấu hình ---
# Danh sách các thư viện phụ trợ. PyTorch sẽ được cài đặt riêng.
LIBRARIES_TO_INSTALL = [
    "PySide6",
    "Pillow",
    "ImageHash",
    "open_clip_torch",
    "tqdm",
    "huggingface_hub",
    "opencv-contrib-python-headless",
    "send2trash",
    "python-dotenv",
    "requests",
    "pandas",
    "matplotlib",
    "seaborn",
    "faiss-cpu"
]

def check_nvidia_gpu():
    """Kiểm tra sự tồn tại của GPU NVIDIA bằng lệnh nvidia-smi."""
    try:
        subprocess.check_output("nvidia-smi", shell=True, stderr=subprocess.DEVNULL)
        print("✅ Đã phát hiện GPU NVIDIA.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Không phát hiện thấy GPU NVIDIA. Sẽ cài đặt phiên bản CPU.")
        return False

def install_pytorch():
    """
    Cài đặt PyTorch, tự động chọn phiên bản GPU hoặc CPU.
    """
    print("\n--- Bắt đầu cài đặt PyTorch ---")
    is_gpu_available = check_nvidia_gpu()
    
    pip_command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]

    if is_gpu_available:
        print("🚀 Cài đặt PyTorch với hỗ trợ CUDA 12.1...")
        pip_command.extend(["--index-url", "https://download.pytorch.org/whl/cu121"])
    else:
        print("⚙️ Cài đặt PyTorch phiên bản CPU...")

    try:
        subprocess.check_call(pip_command)
        print(">>> Cài đặt PyTorch thành công!")
    except subprocess.CalledProcessError as e:
        print("\n❌ LỖI: Không thể cài đặt PyTorch tự động.", file=sys.stderr)
        if is_gpu_available:
            print("Gợi ý: Đảm bảo driver NVIDIA và CUDA Toolkit của bạn đã được cài đặt đúng cách.", file=sys.stderr)
        print("Vui lòng truy cập https://pytorch.org/get-started/locally/ để xem hướng dẫn cài đặt thủ công.", file=sys.stderr)
        print(f"Lỗi chi tiết: {e}", file=sys.stderr)
        sys.exit(1)

def install_other_libraries():
    """Cài đặt các thư viện Python còn lại."""
    print("\n--- Bắt đầu cài đặt các thư viện phụ trợ ---")
    for lib in LIBRARIES_TO_INSTALL:
        print(f"Đang cài đặt {lib}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"✅ Đã cài đặt {lib} thành công.")
        except subprocess.CalledProcessError as e:
            print(f"❌ LỖI: Không thể cài đặt {lib}. Vui lòng thử cài đặt thủ công.", file=sys.stderr)
            print(f"Lỗi chi tiết: {e}", file=sys.stderr)
            sys.exit(1)
    print("--- Hoàn tất cài đặt thư viện phụ trợ ---\n")

# --- Hàm chính ---
if __name__ == "__main__":
    print("======================================================================")
    print("  Chào mừng đến với trình cài đặt độc lập của PixelPure (V26)")
    print("======================================================================")
    
    install_pytorch()
    install_other_libraries()
    
    os.makedirs(os.path.join("core", "models"), exist_ok=True)

    print("\n======================================================================")
    print("🎉 HOÀN TẤT! Môi trường đã được cấu hình thành công.")
    print("Bây giờ bạn có thể chạy 'main.py' để bắt đầu.")
    print("======================================================================")
