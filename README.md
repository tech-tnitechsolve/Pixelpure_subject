# PixelPure - Subject Analysis Tool

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PySide6-GUI-green.svg" alt="PySide6">
  <img src="https://img.shields.io/badge/AI-CLIP_ViT-orange.svg" alt="AI Model">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

## 📖 Giới thiệu

**PixelPure** là công cụ AI tiên tiến được phát triển bởi **TNI Tech Solutions** để phân tích và phát hiện ảnh trùng lặp/tương tự dựa trên Subject Analysis V25+. Ứng dụng sử dụng mô hình CLIP Vision Transformer để nhận diện chủ thể trong ảnh với độ chính xác cao.

### ✨ Tính năng chính

- 🤖 **AI Subject Analysis V25+** - Phân tích chủ thể với độ chính xác 85%+
- 🔍 **Phát hiện ảnh trùng lặp** - Dựa trên hash và cấu trúc
- 🎯 **Nhóm ảnh tương tự** - Theo 4 yếu tố: Subject, Color, Viewpoint, Detail
- 🖼️ **Giao diện hiện đại** - Compact table layout với thumbnail lớn
- 🇻🇳 **Hỗ trợ tiếng Việt** - Giao diện hoàn toàn bằng tiếng Việt
- ⚡ **GPU Accelerated** - Tối ưu cho NVIDIA CUDA
- 📊 **Báo cáo chi tiết** - Thống kê và phân tích kết quả

## 🚀 Cài đặt

### Yêu cầu hệ thống

- **Python 3.8+** (Khuyến nghị Python 3.10)
- **Windows 10/11** 
- **GPU NVIDIA** (tùy chọn, tăng tốc độ xử lý)
- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB
- **Dung lượng**: ~10GB (bao gồm AI models)

### Bước 1: Clone repository

```bash
git clone https://github.com/tech-tnitechsolve/Pixelpure_subject.git
cd Pixelpure_subject
```

### Bước 2: Tạo virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# hoặc
.\.venv\Scripts\activate.bat  # Windows CMD
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Cài đặt PyTorch (tùy chọn GPU)

**Cho GPU NVIDIA:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Cho CPU only:**
```bash
pip install torch torchvision torchaudio
```

## 📋 Dependencies chính

```
PySide6>=6.5.0          # Modern Qt GUI framework
torch>=2.0.0            # PyTorch for AI models  
open-clip-torch>=2.20.0 # CLIP Vision Transformer
opencv-python>=4.8.0    # Computer vision processing
Pillow>=10.0.0          # Image processing
imagehash>=4.3.1        # Perceptual hashing
numpy>=1.24.0           # Numerical computing
```

## 🎮 Sử dụng

### Khởi động ứng dụng

```bash
# Kích hoạt virtual environment
.\.venv\Scripts\Activate.ps1

# Chạy ứng dụng
python app_ui.py

# Hoặc sử dụng launcher (Windows)
double-click "PixelPure Launcher.bat"
```

### Workflow cơ bản

1. **📁 Thêm File**: Click "Thêm File" hoặc "Thêm Thư Mục"
2. **⚙️ Cấu hình**: Điều chỉnh threshold (mặc định 85%)
3. **🔄 Scan**: Click "Bắt Đầu Quét" để phân tích
4. **📊 Xem kết quả**: Review các nhóm ảnh được phát hiện
5. **🗑️ Xử lý**: Xóa hoặc di chuyển ảnh trùng lặp

### Hiểu kết quả phân tích

- **🔄 TRÙNG LẶP 100%**: Ảnh hoàn toàn giống nhau (hash duplicate)
- **🎯 SUBJECT XX%**: Ảnh có cùng chủ thể với độ tương tự XX%
- **❓ KHÔNG XÁC ĐỊNH**: Không thể phân loại chính xác

## 🧠 Công nghệ AI

### Subject Analysis V25+

**PixelPure** sử dụng thuật toán Subject Analysis V25+ với các cải tiến:

- **🎯 Strict Threshold**: 85% thay vì 70% để tránh gộp nhầm
- **🎨 4-Factor Analysis**: 
  - **Subject Detection** (40%): Nhận diện chủ thể chính
  - **Color Analysis** (25%): Phân tích màu sắc dominant
  - **Viewpoint Detection** (20%): Góc nhìn và composition
  - **Detail Recognition** (15%): Mức độ chi tiết và texture

- **⚖️ Penalty System**: Giảm điểm khi có sự khác biệt lớn
- **🔍 Multi-layer Validation**: Nhiều lớp kiểm tra chéo

### AI Models được sử dụng

- **CLIP ViT-H-14**: Mô hình Vision Transformer chính
- **OpenCV ORB**: Phát hiện keypoints và features
- **ImageHash**: Perceptual hashing cho duplicate detection

## 🎨 Giao diện

### Compact Table Layout

- **📋 Status Column**: Hiển thị loại nhóm và similarity score
- **🖼️ Preview Thumbnails**: Thumbnail 144x144px (tăng 1.75x)
- **⚡ Action Panel**: 3 nút chính - Select All, Delete, Move
- **📊 Enhanced Summary**: Thống kê chi tiết với color coding

### Dark Theme

Giao diện tối hiện đại với:
- Material Design principles
- Smooth hover effects
- Vietnamese localization
- Responsive layout

## ⚙️ Cấu hình nâng cao

### Tùy chỉnh threshold

```python
# Trong app_ui.py
self.similarity_threshold = 0.85  # 85% nghiêm ngặt
self.structural_similarity_threshold = 30
```

### GPU Memory Optimization

```python
# Trong scanner.py
self.batch_size = 64 if self.device == "cuda" else 8
torch.backends.cudnn.benchmark = True
```

### Cache Settings

- **Model Cache**: `core/models/` (AI models tự động download)
- **Analysis Cache**: In-memory caching cho tăng tốc
- **ORB Cache**: Feature cache để tái sử dụng

## 🐛 Xử lý sự cố

### Lỗi thường gặp

1. **"Không thể khởi tạo bộ quét"**
   ```bash
   # Kiểm tra virtual environment
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **OutOfMemoryError (GPU)**
   ```python
   # Giảm batch size trong scanner.py
   self.batch_size = 32  # Thay vì 64
   ```

3. **Model download failed**
   ```bash
   # Xóa cache và tải lại
   rm -rf core/models/
   python app_ui.py
   ```

### Performance Tuning

- **RAM < 8GB**: Giảm batch_size xuống 16
- **Slow processing**: Kiểm tra GPU drivers
- **High memory usage**: Đóng các ứng dụng khác

## 📂 Cấu trúc dự án

```
Pixelpure_subject/
├── app_ui.py                 # Main GUI application
├── main.py                   # Entry point
├── install_requirements.py   # Dependency installer
├── requirements.txt          # Python dependencies
├── PixelPure Launcher.bat   # Windows launcher
├── core/
│   ├── scanner.py           # AI scanning engine
│   └── models/              # AI model cache
├── .venv/                   # Virtual environment
└── README.md               # This file
```

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Tạo Pull Request

## 📧 Liên hệ

- **Công ty**: TNI Tech Solutions
- **Email**: support@tnitechsolutions.com
- **Website**: https://tnitechsolutions.com
- **GitHub**: https://github.com/tech-tnitechsolve

## 📄 License

Dự án được phân phối dưới license MIT. Xem `LICENSE` file để biết thêm chi tiết.

## 🙏 Lời cảm ơn

- **OpenAI CLIP** - Foundation model for image understanding
- **Hugging Face** - Model hosting và distribution
- **Qt/PySide6** - Cross-platform GUI framework
- **OpenCV** - Computer vision library

---

<div align="center">
  <strong>Made with ❤️ by TNI Tech Solutions</strong>
  <br>
  <em>Transforming digital asset management with AI</em>
</div>
