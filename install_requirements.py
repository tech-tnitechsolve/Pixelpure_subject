# install_requirements.py
#
# Phiên bản V24: Sửa lỗi Tương thích OpenCV.
# - [CRITICAL-FIX] Thay đổi thư viện `opencv-python-headless` thành
#   `opencv-contrib-python-headless`. Phiên bản "contrib" bao gồm đầy đủ
#   các module, kể cả các thuật toán trích xuất đặc trưng như ORB,
#   giúp giải quyết lỗi "attribute not found" trên một số môi trường.
# - [BENEFIT] Đảm bảo ứng dụng chạy ổn định hơn bất kể phiên bản phụ của OpenCV.

import subprocess
import sys
import os

# --- Cấu hình ---
LIBRARIES_TO_INSTALL = [
    "PySide6",
    "Pillow",
    "ImageHash",
    # "torch" sẽ được cài riêng biệt bên dưới
    "open_clip_torch",
    "tqdm",
    "huggingface_hub",
    "opencv-contrib-python-headless" # [V24] Sử dụng phiên bản contrib để đảm bảo có đủ module
]

def install_pytorch():
    """
    Cài đặt PyTorch, ưu tiên phiên bản GPU nếu có thể.
    """
    print("\n--- Bắt đầu cài đặt PyTorch ---")
    print("Đây là bước quan trọng để ứng dụng có thể chạy trên GPU (nếu có).")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        print(">>> Cài đặt PyTorch thành công!")
    except subprocess.CalledProcessError as e:
        print("LỖI: Không thể cài đặt PyTorch tự động.", file=sys.stderr)
        print("Vui lòng truy cập https://pytorch.org/get-started/locally/ và cài đặt thủ công.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

def install_other_libraries():
    """Cài đặt các thư viện Python còn lại."""
    print("\n--- Bắt đầu cài đặt các thư viện phụ trợ ---")
    for lib in LIBRARIES_TO_INSTALL:
        print(f"Đang cài đặt {lib}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"Đã cài đặt {lib} thành công.")
        except subprocess.CalledProcessError as e:
            print(f"LỖI: Không thể cài đặt {lib}. Vui lòng thử cài đặt thủ công.", file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(1)
    print("--- Hoàn tất cài đặt thư viện phụ trợ ---\n")

# --- Hàm chính ---
if __name__ == "__main__":
    print("Chào mừng đến với trình cài đặt nâng cao của PixelPure (V24)!")
    
    # 1. Cài đặt PyTorch trước
    install_pytorch()
    
    # 2. Cài đặt các thư viện còn lại
    install_other_libraries()
    
    # Tạo thư mục models nếu chưa có
    os.makedirs(os.path.join("core", "models"), exist_ok=True)

    print("======================================================================")
    print("HOÀN TẤT! Môi trường đã được cập nhật để sửa lỗi tương thích OpenCV.")
    print("Bây giờ bạn có thể chạy 'main.py'.")
    print("======================================================================")