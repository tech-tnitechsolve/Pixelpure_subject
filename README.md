# PixelPure (subject)

PixelPure là dự án Python hỗ trợ quét, phân tích và xử lý hình ảnh sử dụng các mô hình AI hiện đại (CLIP, v.v.).

## Tính năng
- Quét, phân tích hình ảnh tự động.
- Sử dụng các mô hình AI mạnh mẽ (CLIP-ViT-H-14, CLIP-ViT-L-14, v.v.).
- Giao diện người dùng đơn giản, trực quan.
- Dashboard kết quả trực tiếp.

## Yêu cầu hệ thống
- Python 3.12 trở lên
- Windows 10/11 (khuyến nghị)
- Kết nối Internet để tải mô hình AI (lần đầu)
- Dung lượng trống: ~5GB cho các mô hình AI (tự động tải về)

## Cài đặt
1. **Clone dự án:**
   ```powershell
   git clone https://github.com/tech-tnitechsolve/Pixelpure_subject.git
   cd Pixelpure_subject
   ```
2. **Tạo môi trường ảo (khuyến nghị):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Cài đặt thư viện:**
   - Chạy script tự động:
     ```powershell
     python install_requirements.py
     ```
   - Hoặc cài đặt thủ công:
     ```powershell
     pip install -r requirements.txt
     ```

**Lưu ý:** Các mô hình AI (CLIP) sẽ được tự động tải về khi chạy lần đầu (~2-3GB).

## Sử dụng
- **Chạy ứng dụng giao diện:**
  ```powershell
  python main.py
  ```
  hoặc chạy file `run.bat` (Windows) hoặc `run.ps1` (PowerShell).

- **Quét và phân tích hình ảnh:**
  1. Mở ứng dụng.
  2. Chọn hình ảnh cần quét.
  3. Nhấn nút "Quét" để bắt đầu phân tích.
  4. Xem kết quả trên giao diện.

## Cấu trúc thư mục
```
Pixelpure_subject/
├── app_ui.py                    # Giao diện người dùng
├── auto_processor.py            # Xử lý tự động
├── cache_manager.py             # Quản lý cache
├── improved_ui_components.py    # Components UI cải tiến
├── install_requirements.py     # Script cài đặt
├── main.py                     # File chính
├── result_dashboard.py         # Dashboard kết quả
├── speed_config.py            # Cấu hình tốc độ
├── requirements.txt           # Dependencies
├── core/
│   └── scanner.py            # Engine quét ảnh
├── .venv/                    # Môi trường ảo (local)
├── __pycache__/             # Cache Python (local)
└── core/models/             # Mô hình AI (tự động tải)
```

**Lưu ý:** Các thư mục có nhãn "(local)" chỉ tồn tại trên máy local và không được đẩy lên Git.

## Đóng góp
- Fork repo, tạo branch mới, commit và gửi pull request.
- Mọi ý kiến đóng góp đều được hoan nghênh!

## Liên hệ
- Email: support@tnitechsolve.com
- Github: [tech-tnitechsolve](https://github.com/tech-tnitechsolve)

---
**Copyright © 2025 TNI Tech Solutions**