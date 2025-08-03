# PixelPure (subject)

PixelPure là một dự án Python giúp quét, phân tích và xử lý hình ảnh sử dụng các mô hình AI hiện đại (CLIP, v.v.).

## Tính năng
- Quét và phân tích hình ảnh.
- Sử dụng các mô hình AI mạnh mẽ (CLIP-ViT-H-14, CLIP-ViT-L-14).
- Giao diện người dùng đơn giản, dễ sử dụng.

## Yêu cầu hệ thống
- Python 3.12 trở lên
- Windows 10/11 (khuyến nghị)
- Kết nối Internet để tải mô hình (lần đầu)

## Cài đặt
1. **Clone dự án:**
   ```powershell
   git clone https://github.com/tech-tnitechsolve/Pixelpure_subject.git
   cd Pixelpure_subject
   ```
2. **Cài đặt thư viện:**
   - Chạy script tự động:
     ```powershell
     python install_requirements.py
     ```
   - Hoặc cài đặt thủ công:
     ```powershell
     pip install -r requirements.txt
     ```

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
├── app_ui.py
├── main.py
├── install_requirements.py
├── run.bat
├── run.ps1
├── core/
│   ├── scanner.py
│   └── models/
│       └── ... (các mô hình AI)
└── test_app.py
```

## Đóng góp
- Fork repo, tạo branch mới, commit và gửi pull request.
- Mọi ý kiến đóng góp đều được hoan nghênh!

## Liên hệ
- Email: support@tnitechsolve.com
- Github: [tech-tnitechsolve](https://github.com/tech-tnitechsolve)

---
**Copyright © 2025 TNI Tech Solutions**
