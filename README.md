# 🖼️ PixelPure - AI-Powered Image Analysis & Cleanup Tool

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/tech-tnitechsolve/Pixelpure_subject?style=for-the-badge)](https://github.com/tech-tnitechsolve/Pixelpure_subject/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/tech-tnitechsolve/Pixelpure_subject?style=for-the-badge)](https://github.com/tech-tnitechsolve/Pixelpure_subject/issues)
[![GitHub License](https://img.shields.io/github/license/tech-tnitechsolve/Pixelpure_subject?style=for-the-badge)](https://github.com/tech-tnitechsolve/Pixelpure_subject/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue?style=for-the-badge)](https://www.python.org/downloads/)

**PixelPure** là công cụ AI thông minh để **quét, phân tích và dọn dẹp hình ảnh** sử dụng mô hình CLIP hiện đại. Giúp bạn tự động phát hiện và xử lý ảnh trùng lặp, tương tự với độ chính xác cao.

[🚀 Demo](#-demo) • [📖 Cài đặt](#-cài-đặt) • [💡 Tính năng](#-tính-năng) • [🔧 Sử dụng](#-sử-dụng) • [🤝 Đóng góp](#-đóng-góp)

</div>

---

## 🌟 Tính năng

### 🎯 **Core Features**
- **🔍 AI Image Scanner:** Sử dụng mô hình CLIP-ViT-H-14 để phân tích hình ảnh với độ chính xác cao
- **🧹 Smart Cleanup:** Tự động phát hiện và xử lý ảnh trùng lặp, tương tự  
- **⚡ 8-Layer Detection:** Hệ thống quét 8 tầng với anti-mistake technology
- **🎨 Modern UI:** Giao diện đẹp, trực quan với drag-and-drop support

### 🛠️ **Advanced Features**  
- **📊 Enhanced Dashboard:** Kết quả phân tích chi tiết với thống kê trực quan
- **🔄 Auto-Processing:** Xử lý tự động với business rules thông minh
- **📈 Performance Optimized:** Hỗ trợ GPU acceleration và batch processing
- **💾 Smart Caching:** Cache thông minh để tăng tốc độ xử lý

### 🎪 **Detection Types**
- **🔄 Duplicates:** Phát hiện ảnh trùng lặp 100% (giữ file lớn nhất)
- **🎯 Similar:** Phát hiện ảnh tương tự (đổi tên theo pattern 1(1), 1(2)...)  
- **🧬 Hybrid:** Nhóm hỗn hợp (xóa trùng lặp + đổi tên tương tự)

---

## 📋 Yêu cầu hệ thống

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11, macOS 10.15+, Linux Ubuntu 18.04+ |
| **Python** | 3.12+ (khuyến nghị 3.12.0) |
| **RAM** | 8GB+ (16GB khuyến nghị cho datasets lớn) |
| **Storage** | 5GB+ (cho AI models) |
| **GPU** | Optional: CUDA-compatible GPU (tăng tốc 10x) |
| **Internet** | Cần kết nối cho lần đầu tải models |

---

## 🚀 Cài đặt

### **Phương pháp 1: Quick Install (Khuyến nghị)**

```bash
# 1. Clone repository
git clone https://github.com/tech-tnitechsolve/Pixelpure_subject.git
cd Pixelpure_subject

# 2. Tạo môi trường ảo
python -m venv .venv

# 3. Kích hoạt môi trường ảo
# Windows
.venv\Scripts\activate
# macOS/Linux  
source .venv/bin/activate

# 4. Cài đặt tự động
python install_requirements.py
```

### **Phương pháp 2: Manual Install**

```bash
# Cài đặt thủ công từ requirements.txt
pip install -r requirements.txt

# Hoặc cài đặt từng package chính
pip install torch torchvision torchaudio
pip install open-clip-torch
pip install PySide6
pip install Pillow opencv-python imagehash
pip install send2trash
```

### **🔥 One-liner cho Windows:**
```powershell
git clone https://github.com/tech-tnitechsolve/Pixelpure_subject.git && cd Pixelpure_subject && python -m venv .venv && .venv\Scripts\activate && python install_requirements.py
```

---

## 🔧 Sử dụng

### **🖥️ Chạy ứng dụng**
```bash
python main.py
```

### **📋 Workflow cơ bản**

1. **🎯 Chọn ảnh:**
   - Drag & drop files/folders vào ứng dụng
   - Hoặc click để chọn từ file explorer

2. **⚙️ Cấu hình:**
   - Chọn files cần scan (có thể select all)
   - Kiểm tra AI model đã sẵn sàng ✅

3. **🔍 Quét và phân tích:**
   - Nhấn "Bắt đầu quét" 
   - Theo dõi tiến trình 8-layer scanning

4. **🎪 Xem kết quả:**
   - Xem phân loại: Duplicates, Similar, Hybrid
   - Preview các nhóm ảnh được detect

5. **⚡ Xử lý tự động:**
   - Nhấn "Tự động xử lý" 
   - Chọn "Có" để rescan files còn lại

### **🎨 Screenshots**

<details>
<summary>Click để xem screenshots</summary>

```
🏠 Main Interface          📊 Scan Results           🎯 Auto Processing
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  📁 Drop Zone   │  -->  │ 🔄 2 Duplicates │  -->  │ ⚡ Processing   │
│  Click/Drag     │       │ 🎯 5 Similar    │       │ 📋 Results      │
│  Files Here     │       │ 🧬 3 Hybrid     │       │ 🔄 Rescan?      │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

</details>

---

## 📁 Cấu trúc dự án

```
Pixelpure_subject/
├── 📁 core/
│   ├── scanner.py              # 🔧 AI scanning engine (8-layer)
│   └── models/                 # 🤖 Auto-downloaded AI models
├── 📄 app_ui.py               # 🎨 Main UI components  
├── 📄 main.py                 # 🚀 Entry point
├── 📄 auto_processor.py       # ⚡ Auto-processing logic
├── 📄 cache_manager.py        # 💾 Smart caching system
├── 📄 speed_config.py         # ⚙️ Performance configuration
├── 📄 result_dashboard.py     # 📊 Results visualization
├── 📄 improved_ui_components.py # 🎪 Enhanced UI widgets
├── 📄 install_requirements.py # 🔧 Auto-installer script
├── 📄 requirements.txt        # 📋 Dependencies
├── 📄 README.md              # 📖 This file
├── 📄 .gitignore             # 🚫 Git ignore rules
└── 📁 .venv/                 # 🐍 Virtual environment (local)
```

---

## 🔬 Công nghệ sử dụng

| Category | Technologies |
|----------|-------------|
| **🤖 AI/ML** | PyTorch, OpenCLIP, CLIP-ViT-H-14, ImageHash |
| **🎨 UI/UX** | PySide6 (Qt6), Custom Components |
| **🖼️ Image** | Pillow (PIL), OpenCV, NumPy |
| **⚡ Performance** | CUDA, Batch Processing, Smart Caching |
| **🔧 Utils** | Send2Trash, OS Integration |

---

## 🎯 Roadmap

- [x] ✅ **v1.0:** Core scanning với CLIP
- [x] ✅ **v1.1:** 8-layer anti-mistake detection  
- [x] ✅ **v1.2:** Enhanced UI với auto-processing
- [x] ✅ **v1.3:** Sequential group numbering
- [ ] 🔄 **v1.4:** Batch API cho automation
- [ ] 🔄 **v1.5:** Web interface  
- [ ] 🔄 **v2.0:** Multi-model support (DINOV2, etc.)

---

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! 

### **🔧 Development Setup**
```bash
# 1. Fork repository trên GitHub
# 2. Clone fork của bạn
git clone https://github.com/YOUR_USERNAME/Pixelpure_subject.git

# 3. Tạo branch mới
git checkout -b feature/your-feature-name

# 4. Make changes và test
python main.py

# 5. Commit và push
git commit -m "Add: your feature description"
git push origin feature/your-feature-name

# 6. Tạo Pull Request
```

### **🐛 Bug Reports**
- Sử dụng [GitHub Issues](https://github.com/tech-tnitechsolve/Pixelpure_subject/issues)
- Bao gồm: OS, Python version, error logs, steps to reproduce

### **💡 Feature Requests**  
- Mô tả chi tiết feature mong muốn
- Giải thích use case và benefit
- Attach mockups nếu có

---

## 📝 License

Dự án được phát hành dưới [MIT License](LICENSE).

```
MIT License - Copyright (c) 2025 TNI Tech Solutions
Permission is hereby granted, free of charge, to any person obtaining a copy...
```

---

## 📞 Liên hệ & Hỗ trợ

<div align="center">

| Contact Method | Information |
|----------------|-------------|
| **📧 Email** | [support@tnitechsolve.com](mailto:support@tnitechsolve.com) |
| **🐙 GitHub** | [@tech-tnitechsolve](https://github.com/tech-tnitechsolve) |
| **🌐 Website** | [tnitechsolve.com](https://tnitechsolve.com) |
| **💬 Issues** | [GitHub Issues](https://github.com/tech-tnitechsolve/Pixelpure_subject/issues) |

---

<sub>**🌟 Nếu project hữu ích, đừng quên để lại ⭐ star trên GitHub!**</sub>

**Made with ❤️ by TNI Tech Solutions**

</div>

---

<details>
<summary>📊 Statistics</summary>

```
📈 Project Stats (Updated: August 2025)
├── 🗂️ Languages: Python (95%), Shell (5%)  
├── 📝 Lines of Code: 2000+
├── 🔧 Dependencies: 15+ packages
├── 🤖 AI Models: CLIP-ViT-H-14 (3.94GB)
├── ⚡ Performance: 10K+ images support
└── 🎯 Accuracy: 92%+ similarity detection
```

</details>