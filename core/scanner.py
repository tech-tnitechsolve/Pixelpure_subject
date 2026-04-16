# core/scanner.py
#
# Phiên bản V25+: Subject Analysis Nghiêm Ngặt - Tránh gộp nhầm hoàn toàn.
# - [FIXED] Tăng threshold lên 85% thay vì 70% để tránh gộp các Subject khác nhau
# - [ENHANCED] Penalty system cho màu sắc và viewpoint khác biệt quá nhiều
# - [STRICT] Multiple validation layers: Subject 80%+ Color 75%+ Viewpoint 65%+
# - [OPTIMIZED] Trọng số mới: Color 45% (quan trọng nhất) + Viewpoint 25% + Detail 15% + Material 15%

import os
import numpy as np
from typing import Optional, Callable, cast, List, Dict, Any, Tuple
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PIL import Image, UnidentifiedImageError, ImageStat
import imagehash
import torch
from torch import Tensor
import open_clip
from open_clip import CLIP
import cv2
from collections import defaultdict
import gc

class DSU:
    """Lớp hỗ trợ cho thuật toán Disjoint Set Union để gom nhóm hiệu quả."""
    def __init__(self, items):
        self.parent = {item: item for item in items}
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i

class ScannerWorker(QThread):
    progress_updated = Signal(int, int, str)
    scan_completed = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, file_list, similarity_threshold, speed_mode='fast', allow_cpu_fallback=False):
        super().__init__()
        self.file_list = file_list
        self.allow_cpu_fallback = allow_cpu_fallback
        
        # THÊM: Import cấu hình tốc độ
        from speed_config import SpeedConfig
        self.speed_config = SpeedConfig.get_mode(speed_mode)
        
        # Áp dụng cấu hình tốc độ
        self.similarity_threshold = self.speed_config['similarity_threshold']
        self.structural_similarity_threshold = 30  # Tăng ngưỡng cấu trúc
        self.is_running = True
        self.mutex = QMutex()
        self.pause_condition = QWaitCondition()
        self._is_paused = False
        self.model: Optional[CLIP] = None
        self.preprocess: Optional[Callable] = None
        self.tokenizer: Optional[Callable] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Batch size theo cấu hình
        self.batch_size = (self.speed_config['batch_size_gpu'] if self.device == "cuda" 
                          else self.speed_config['batch_size_cpu'])
        
        self.orb_cache: Dict[str, Any] = {}
        # Cache cho 4 yếu tố phân tích Subject
        self.subject_analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # Các tùy chọn tốc độ
        self.fast_mode = self.speed_config['fast_mode']
        self.parallel_processing = self.speed_config['parallel_processing']
        self.skip_detailed_analysis = self.speed_config.get('skip_detailed_analysis', False)
        
        # Thread pool cho xử lý song song
        if self.parallel_processing:
            from concurrent.futures import ThreadPoolExecutor
            self.thread_pool = ThreadPoolExecutor(max_workers=self.speed_config['max_workers'])
        else:
            self.thread_pool = None

    def _load_model(self):
        model_name = self.speed_config['model_name']
        pretrained_tag = self.speed_config['model_pretrained']
        
        self.progress_updated.emit(0, 0, f"🤖 Đang tải AI Model: {model_name}")
        
        try:
            # Tối ưu GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
            
            # Check if model exists
            model_cache_dir = os.path.join("core", "models")
            model_exists = self._check_model_exists(model_name, pretrained_tag, model_cache_dir)
            
            if not model_exists:
                self.progress_updated.emit(0, 0, f"📥 Đang tải xuống model {model_name} lần đầu...")
                self.progress_updated.emit(0, 0, f"⏳ Vui lòng chờ, model có thể lớn (500MB-4GB)...")
            else:
                self.progress_updated.emit(0, 0, f"✅ Sử dụng model đã có: {model_name}")
            
            # Add cache directory to manager
            from cache_manager import cache_manager
            cache_manager.add_cache_dir(model_cache_dir)
                
            model_obj, _, preprocess_fn = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained_tag,
                device=self.device,
                cache_dir=model_cache_dir
            )
            
            self.progress_updated.emit(0, 0, f"🔧 Đang tối ưu model cho {self.device.upper()}...")
            
            self.model = cast(CLIP, model_obj)
            self.preprocess = cast(Callable, preprocess_fn)
            self.tokenizer = open_clip.get_tokenizer(model_name)
            
            if self.model: 
                self.model.eval()
                # Tối ưu memory với half precision trên GPU
                if self.device == "cuda":
                    self.model = self.model.half()
            
            self.progress_updated.emit(0, 0, f"✅ Model {model_name} sẵn sàng trên {self.device.upper()}")
            print(f"INFO: AI Model {model_name} optimized for SPEED on {self.device.upper()}")
            
        except Exception as e:
            raise RuntimeError(f"Không thể tải AI Model {model_name}.\nLỗi: {e}")
    
    def _check_model_exists(self, model_name: str, pretrained_tag: str, cache_dir: str) -> bool:
        """Kiểm tra xem model đã được download chưa"""
        try:
            # Check common model file patterns
            model_patterns = [
                f"*{model_name}*{pretrained_tag}*",
                f"*{model_name}*",
                "*.safetensors",
                "*.bin",
                "*.pt"
            ]
            
            if os.path.exists(cache_dir):
                for root, dirs, files in os.walk(cache_dir):
                    for file in files:
                        if any(pattern.replace('*', '') in file.lower() for pattern in model_patterns):
                            return True
            return False
        except:
            return False

    def _get_file_info(self, file_path):
        try:
            with Image.open(file_path) as img:
                img.load()
                return {
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "width": img.width,
                    "height": img.height
                }
        except (UnidentifiedImageError, FileNotFoundError, PermissionError, SyntaxError, OSError) as e:
            print(f"[WARNING] Lỗi khi đọc file ảnh \"{file_path}\": {e}")
            return None  # Bỏ qua file ảnh bị hỏng hoặc không đọc được
        except Exception as e:
            # Bắt mọi lỗi khác liên quan đến file ảnh, ví dụ PNG bị hỏng
            print(f"[WARNING] File ảnh bị lỗi hoặc không hợp lệ (bỏ qua): {file_path} | Lỗi: {e}")
            return None

    def _analyze_subject_detail(self, file_path: str) -> float:
        """Phân tích mức độ chi tiết của Subject - TĂNG TỐC"""
        if file_path in self.subject_analysis_cache and 'detail' in self.subject_analysis_cache[file_path]:
            return self.subject_analysis_cache[file_path]['detail']
        
        try:
            # TĂNG TỐC: Resize ảnh nhỏ hơn để xử lý nhanh
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return 0.0
            
            # Resize xuống 256x256 thay vì full size
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            
            if self.fast_mode:
                # CHẾ ĐỘ NHANH: Chỉ dùng Laplacian
                laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                detail_score = min(1.0, laplacian_var / 1000.0)
            else:
                # CHẾ ĐỘ CHẬM: Dùng cả Laplacian và Canny
                laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                edges = cv2.Canny(img, 50, 150)
                edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
                detail_score = min(1.0, (laplacian_var / 1000.0 + edge_density * 10) / 2)
            
            if file_path not in self.subject_analysis_cache:
                self.subject_analysis_cache[file_path] = {}
            self.subject_analysis_cache[file_path]['detail'] = detail_score
            return detail_score
        except Exception:
            return 0.0

    def _analyze_subject_color(self, file_path: str) -> np.ndarray:
        """Phân tích phổ màu chủ đạo của Subject - TĂNG TỐC"""
        if file_path in self.subject_analysis_cache and 'color' in self.subject_analysis_cache[file_path]:
            return self.subject_analysis_cache[file_path]['color']
        
        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                
                if self.fast_mode:
                    # CHẾ ĐỘ NHANH: Resize nhỏ hơn và ít bins hơn
                    img = img.resize((64, 64), Image.Resampling.NEAREST)  # Nhanh hơn LANCZOS
                    bins = 4  # Giảm từ 8 xuống 4
                else:
                    # CHẾ ĐỘ CHẬM: Resize lớn hơn và nhiều bins
                    img = img.resize((128, 128), Image.Resampling.LANCZOS)
                    bins = 8
                
                img_array = np.array(img)
                
                # Tính histogram RGB với bins ít hơn
                r_hist = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256))[0]
                g_hist = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256))[0]
                b_hist = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256))[0]
                
                # Normalize
                total_pixels = img.size[0] * img.size[1]
                color_vector = np.concatenate([r_hist, g_hist, b_hist]) / total_pixels
                
                if file_path not in self.subject_analysis_cache:
                    self.subject_analysis_cache[file_path] = {}
                self.subject_analysis_cache[file_path]['color'] = color_vector
                return color_vector
        except Exception:
            bins = 4 if self.fast_mode else 8
            return np.zeros(bins * 3)  # bins*3 for RGB

    def _analyze_subject_viewpoint(self, file_path: str) -> np.ndarray:
        """Phân tích góc nhìn và composition của Subject - TĂNG TỐC"""
        if file_path in self.subject_analysis_cache and 'viewpoint' in self.subject_analysis_cache[file_path]:
            return self.subject_analysis_cache[file_path]['viewpoint']
        
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                grid_size = 2 if self.fast_mode else 3
                return np.zeros(grid_size * grid_size)
            
            # TĂNG TỐC: Resize ảnh nhỏ hơn
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            
            if self.fast_mode:
                # CHẾ ĐỘ NHANH: Lưới 2x2 thay vì 3x3
                grid_size = 2
            else:
                # CHẾ ĐỘ CHẬM: Lưới 3x3
                grid_size = 3
            
            # Chia ảnh thành lưới để phân tích composition
            h, w = img.shape
            grid_features = []
            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * h // grid_size, (i + 1) * h // grid_size
                    x1, x2 = j * w // grid_size, (j + 1) * w // grid_size
                    grid_region = img[y1:y2, x1:x2]
                    # Tính intensity trung bình cho mỗi region
                    avg_intensity = float(np.mean(grid_region.astype(np.float32))) / 255.0
                    grid_features.append(avg_intensity)
            
            viewpoint_vector = np.array(grid_features)
            
            if file_path not in self.subject_analysis_cache:
                self.subject_analysis_cache[file_path] = {}
            self.subject_analysis_cache[file_path]['viewpoint'] = viewpoint_vector
            return viewpoint_vector
        except Exception:
            grid_size = 2 if self.fast_mode else 3
            return np.zeros(grid_size * grid_size)

    def _analyze_subject_material(self, file_path: str) -> np.ndarray:
        """Phân tích texture và material của Subject - TĂNG TỐC"""
        if file_path in self.subject_analysis_cache and 'material' in self.subject_analysis_cache[file_path]:
            return self.subject_analysis_cache[file_path]['material']
        
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                feature_count = 3 if self.fast_mode else 6
                return np.zeros(feature_count)
            
            # TĂNG TỐC: Resize nhỏ hơn
            resize_size = 64 if self.fast_mode else 128
            img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
            
            if self.fast_mode:
                # CHẾ ĐỘ NHANH: Chỉ 3 features cơ bản
                features = [
                    float(np.var(img.astype(np.float32))) / 10000.0,  # Variance
                    float(np.std(img.astype(np.float32))) / 255.0,    # Standard deviation  
                    len(np.unique(img)) / 256.0,                      # Unique values ratio
                ]
            else:
                # CHẾ ĐỘ CHẬM: 6 features đầy đủ
                sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                
                features = [
                    float(np.var(img.astype(np.float32))) / 10000.0,  # Variance
                    float(np.mean(np.abs(sobel_x))) / 255.0,          # Horizontal edges
                    float(np.mean(np.abs(sobel_y))) / 255.0,          # Vertical edges
                    float(np.std(img.astype(np.float32))) / 255.0,    # Standard deviation
                    len(np.unique(img)) / 256.0,                      # Unique values ratio
                    float(np.mean(cv2.Laplacian(img, cv2.CV_64F).astype(np.float32))) / 255.0  # Smoothness
                ]
            
            material_vector = np.array(features)
            
            if file_path not in self.subject_analysis_cache:
                self.subject_analysis_cache[file_path] = {}
            self.subject_analysis_cache[file_path]['material'] = material_vector
            return material_vector
        except Exception:
            feature_count = 3 if self.fast_mode else 6
            return np.zeros(feature_count)

    def _get_or_compute_subject_features(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Tính toán đặc trưng Subject tổng hợp từ 4 yếu tố"""
        if file_path in self.orb_cache:
            return self.orb_cache[file_path]
        
        try:
            # Thu thập 4 yếu tố Subject
            detail_score = self._analyze_subject_detail(file_path)
            color_vector = self._analyze_subject_color(file_path)
            viewpoint_vector = self._analyze_subject_viewpoint(file_path)
            material_vector = self._analyze_subject_material(file_path)
            
            # Kết hợp thành vector đặc trưng Subject
            subject_features = {
                'detail': detail_score,
                'color': color_vector,
                'viewpoint': viewpoint_vector,  
                'material': material_vector,
                'combined': np.concatenate([
                    [detail_score],
                    color_vector,
                    viewpoint_vector,
                    material_vector
                ])
            }
            
            self.orb_cache[file_path] = subject_features
            return subject_features
        except Exception:
            return None

    def _calculate_subject_similarity(self, path1: str, path2: str) -> float:
        """Tính toán độ tương đồng Subject nghiêm ngặt để tránh gộp nhầm"""
        features1 = self._get_or_compute_subject_features(path1)
        features2 = self._get_or_compute_subject_features(path2)

        if features1 is None or features2 is None:
            return 0.0
        
        try:
            # TRỌNG SỐ MỚI - ưu tiên màu sắc và viewpoint để phân biệt Subject
            weights = {
                'detail': 0.15,    # Giảm detail vì có thể giống nhau
                'color': 0.45,     # TĂNG màu sắc - quan trọng nhất để phân biệt Subject
                'viewpoint': 0.25, # Tăng viewpoint - composition khác nhau
                'material': 0.15   # Giảm material vì có thể tương tự
            }
            
            # Tính similarity cho từng yếu tố
            detail_sim = 1.0 - abs(features1['detail'] - features2['detail'])
            
            # NGHIÊM NGẶT HÓA Color similarity 
            color_sim = np.dot(features1['color'], features2['color']) / (
                np.linalg.norm(features1['color']) * np.linalg.norm(features2['color']) + 1e-8
            )
            # PENALTY nếu color quá khác biệt
            if color_sim < 0.6:  # Nếu màu sắc khác biệt > 40%
                color_sim = color_sim * 0.5  # Penalty mạnh
            
            # NGHIÊM NGẶT HÓA Viewpoint similarity
            viewpoint_sim = np.dot(features1['viewpoint'], features2['viewpoint']) / (
                np.linalg.norm(features1['viewpoint']) * np.linalg.norm(features2['viewpoint']) + 1e-8
            )
            # PENALTY nếu composition quá khác
            if viewpoint_sim < 0.5:
                viewpoint_sim = viewpoint_sim * 0.3
            
            # Material similarity với penalty
            material_sim = np.dot(features1['material'], features2['material']) / (
                np.linalg.norm(features1['material']) * np.linalg.norm(features2['material']) + 1e-8
            )
            
            # Tổng hợp với trọng số và PENALTY SYSTEM
            total_similarity = (
                weights['detail'] * detail_sim +
                weights['color'] * color_sim +
                weights['viewpoint'] * viewpoint_sim +
                weights['material'] * material_sim
            )
            
            # THÊM MULTIPLE PENALTIES để tránh gộp nhầm
            penalties = 0
            
            # Penalty 1: Nếu màu sắc quá khác biệt
            if color_sim < 0.7:
                penalties += 0.2
            
            # Penalty 2: Nếu viewpoint quá khác biệt  
            if viewpoint_sim < 0.6:
                penalties += 0.15
                
            # Penalty 3: Nếu material quá khác biệt
            if material_sim < 0.5:
                penalties += 0.1
            
            # Áp dụng penalties
            final_similarity = max(0.0, total_similarity - penalties)
            
            return min(1.0, final_similarity)
        except Exception:
            return 0.0

    def _check_pause(self):
        self.mutex.lock()
        while self._is_paused:
            self.pause_condition.wait(self.mutex)
        self.mutex.unlock()

    def run(self):
        try:
            if len(self.file_list) < 2:
                self.scan_completed.emit([]); return

            # Enforce GPU-only mode if required by speed config
            if self.speed_config.get('use_gpu_only', False) and self.device != "cuda":
                if not self.allow_cpu_fallback:
                    self.error_occurred.emit("Chế độ Chất Lượng Cao yêu cầu GPU (CUDA). Bạn có muốn chạy bằng CPU không? (Chậm hơn nhiều)")
                    self.scan_completed.emit([])
                    return
                else:
                    # Override to CPU explicitly for clarity
                    self.device = "cpu"

            self._load_model()
            if not self.model or not self.preprocess or not self.tokenizer:
                self.error_occurred.emit("AI Model chưa được tải cho Subject Analysis."); return

            # Tầng 1: Lọc Trùng lặp Tuyệt đối bằng Multi-Hash Analysis
            total_files = len(self.file_list)
            file_hashes = {}  # file_path -> {'dhash': hash, 'phash': hash, 'ahash': hash, 'size': int, 'dimensions': tuple}
            all_file_info = {}
            self.progress_updated.emit(0, total_files, f"Tầng 1/5: Phát hiện ảnh trùng lặp (Multi-Hash)...")
            
            for idx, file_path in enumerate(self.file_list):
                self._check_pause();
                if not self.is_running: return
                self.progress_updated.emit(idx + 1, total_files, f"Tầng 1/5: Multi-Hash ảnh ({idx+1}/{total_files})")
                info = self._get_file_info(file_path)
                if not info: continue
                all_file_info[file_path] = info
                try:
                    with Image.open(file_path) as img:
                        img_rgb = img.convert("RGB")
                        # Tính 3 loại hash khác nhau để kiểm tra chéo
                        dhash_val = imagehash.dhash(img_rgb)
                        phash_val = imagehash.phash(img_rgb)
                        ahash_val = imagehash.average_hash(img_rgb)
                        
                        file_hashes[file_path] = {
                            'dhash': dhash_val,
                            'phash': phash_val, 
                            'ahash': ahash_val,
                            'size': info['size'],
                            'dimensions': (info['width'], info['height'])
                        }
                except Exception: continue
            
            # Tìm ảnh trùng lặp thực sự bằng multi-hash validation
            duplicate_groups = []
            files_for_deep_scan = []
            processed_files = set()
            
            for file1 in file_hashes:
                if file1 in processed_files: continue
                
                duplicate_candidates = [file1]
                hash1 = file_hashes[file1]
                
                for file2 in file_hashes:
                    if file2 == file1 or file2 in processed_files: continue
                    hash2 = file_hashes[file2]
                    
                    # KIỂM TRA NGHIÊM NGẶT: Phải thỏa mãn TẤT CẢ điều kiện sau
                    is_true_duplicate = (
                        # 1. Ít nhất 2/3 hash phải giống nhau hoàn toàn (distance = 0)
                        sum([
                            hash1['dhash'] - hash2['dhash'] == 0,
                            hash1['phash'] - hash2['phash'] == 0, 
                            hash1['ahash'] - hash2['ahash'] == 0
                        ]) >= 2 and
                        
                        # 2. Kích thước file gần giống nhau (chênh lệch < 5% hoặc < 10KB)
                        (abs(hash1['size'] - hash2['size']) < max(hash1['size'] * 0.05, 10240)) and
                        
                        # 3. Kích thước ảnh phải giống nhau hoàn toàn
                        hash1['dimensions'] == hash2['dimensions']
                    )
                    
                    if is_true_duplicate:
                        duplicate_candidates.append(file2)
                        processed_files.add(file2)
                
                processed_files.add(file1)
                
                if len(duplicate_candidates) > 1:
                    # Đây là nhóm trùng lặp thực sự
                    duplicate_groups.append({ 
                        "type": "duplicate", 
                        "files": [all_file_info[p] for p in duplicate_candidates if p in all_file_info], 
                        "score": 1.0,
                        "analysis_method": "Multi-Hash Validation (dhash+phash+ahash+size+dimensions)"
                    })
                else:
                    # File này không trùng lặp, đưa vào deep scan
                    files_for_deep_scan.append(file1)
            
            # Tầng 2: Phân tích Subject với xử lý SONG SONG
            num_deep_scan = len(files_for_deep_scan)
            self.progress_updated.emit(0, num_deep_scan, "Tầng 2/5: Phân tích Subject SONG SONG...")
            
            subject_features = {}
            if self.fast_mode and num_deep_scan > 50:
                # XỬ LÝ SONG SONG cho file nhiều
                from concurrent.futures import as_completed
                
                def process_file_batch(file_batch):
                    batch_results = {}
                    for file_path in file_batch:
                        if not self.is_running: break
                        features = self._get_or_compute_subject_features(file_path)
                        if features:
                            batch_results[file_path] = features
                    return batch_results
                
                # Chia thành các batch nhỏ để xử lý song song
                batch_size = 20
                file_batches = [files_for_deep_scan[i:i+batch_size] 
                              for i in range(0, len(files_for_deep_scan), batch_size)]
                
                futures = []
                for batch in file_batches:
                    future = self.thread_pool.submit(process_file_batch, batch)
                    futures.append(future)
                
                processed = 0
                for future in as_completed(futures):
                    if not self.is_running: break
                    batch_results = future.result()
                    subject_features.update(batch_results)
                    processed += len(batch_results)
                    self.progress_updated.emit(processed, num_deep_scan, 
                                             f"Tầng 2/5: Song song ({processed}/{num_deep_scan})")
            else:
                # XỬ LÝ TUẦN TỰ cho file ít
                for idx, file_path in enumerate(files_for_deep_scan):
                    self._check_pause();
                    if not self.is_running: return
                    self.progress_updated.emit(idx + 1, num_deep_scan, f"Tầng 2/5: Phân tích Subject ({idx+1}/{num_deep_scan})")
                    features = self._get_or_compute_subject_features(file_path)
                    if features:
                        subject_features[file_path] = features

            # Tầng 3: Trích xuất AI Features (GPU Optimized Batch Processing)
            self.progress_updated.emit(0, num_deep_scan, "Tầng 3/5: AI Feature Extraction...")
            ai_features = {}
            
            # Tối ưu GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                if self.device == "cuda":
                    # Sử dụng autocast tương thích với phiên bản PyTorch hiện tại
                    try:
                        # Thử sử dụng torch.amp.autocast mới (PyTorch 2.0+)
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            ai_features = self._extract_ai_features_batch(files_for_deep_scan, num_deep_scan)
                    except (AttributeError, TypeError):
                        # Fallback về torch.cuda.amp.autocast (PyTorch < 2.0)
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FutureWarning)
                            with torch.cuda.amp.autocast():
                                ai_features = self._extract_ai_features_batch(files_for_deep_scan, num_deep_scan)
                else:
                    ai_features = self._extract_ai_features_batch(files_for_deep_scan, num_deep_scan)

            if not ai_features:
                self.scan_completed.emit(duplicate_groups); return

            # Tầng 4: Subject Clustering nghiêm ngặt với threshold 85%
            num_features = len(subject_features)
            self.progress_updated.emit(0, num_features, "Tầng 4/5: Subject Clustering nghiêm ngặt (85% threshold)...")
            
            subject_groups = self._perform_subject_clustering(subject_features, ai_features)
            
            # Tầng 5: Super Strict Validation
            self.progress_updated.emit(0, len(subject_groups), "Tầng 5/5: Super Strict Validation...")
            final_groups = self._validate_and_score_groups(subject_groups, all_file_info)

            # Kết hợp và sắp xếp kết quả
            all_final_groups = duplicate_groups + final_groups
            all_final_groups.sort(key=lambda g: g.get('score', 0.0), reverse=True)
            
            # Dọn dẹp memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            self.scan_completed.emit(all_final_groups)

        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Lỗi trong Subject Analysis:\n{e}\n{traceback.format_exc()}")

    def _extract_ai_features_batch(self, file_paths: List[str], total_files: int) -> Dict[str, Tensor]:
        """Trích xuất AI features với batch processing SIÊU TỐI ƯU"""
        ai_features = {}
        
        # TĂNG TỐC: Preload nhiều batch cùng lúc
        for i in range(0, len(file_paths), self.batch_size):
            self._check_pause();
            if not self.is_running: return {}
            
            batch_paths = file_paths[i:i+self.batch_size]
            batch_images, valid_paths_in_batch = [], []
            
            # TĂNG TỐC: Xử lý song song việc load ảnh
            def load_single_image(file_path):
                try:
                    with Image.open(file_path) as img:
                        if self.fast_mode:
                            # CHẾ ĐỘ NHANH: Resize nhỏ hơn trước khi preprocess
                            img = img.resize((224, 224), Image.Resampling.BILINEAR)
                        
                        if self.preprocess is not None:
                            image_input = self.preprocess(img).to(self.device)
                            if self.device == "cuda":
                                image_input = image_input.half()
                            return image_input, file_path
                except Exception: 
                    pass
                return None, None
            
            # Load ảnh song song
            if len(batch_paths) > 10 and self.fast_mode:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(load_single_image, path) for path in batch_paths]
                    for future in as_completed(futures):
                        image_input, file_path = future.result()
                        if image_input is not None and file_path is not None:
                            batch_images.append(image_input)
                            valid_paths_in_batch.append(file_path)
            else:
                # Load tuần tự cho batch nhỏ
                for file_path in batch_paths:
                    image_input, valid_path = load_single_image(file_path)
                    if image_input is not None and valid_path is not None:
                        batch_images.append(image_input)
                        valid_paths_in_batch.append(valid_path)
            
            if not batch_images: continue
            
            self.progress_updated.emit(
                i + len(batch_paths), total_files, 
                f"Tầng 3/5: AI Batch {(i//self.batch_size)+1} ({i+len(batch_paths)}/{total_files})"
            )
            
            # TĂNG TỐC: Xử lý batch với memory optimization
            batch_tensor = torch.stack(batch_images)
            if self.model is not None:
                with torch.no_grad():  # Đảm bảo không tính gradient
                    image_features = self.model.encode_image(batch_tensor)
                    
                    if image_features is not None:
                        image_features = image_features.float()  # Convert back to float32
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        for path, feature_vec in zip(valid_paths_in_batch, image_features):
                            ai_features[path] = feature_vec.unsqueeze(0)
            
            # TĂNG TỐC: Giải phóng memory ngay
            del batch_tensor, batch_images
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return ai_features

    def _perform_subject_clustering(self, subject_features: Dict[str, Dict], ai_features: Dict[str, Tensor]) -> List[List[str]]:
        """Thực hiện clustering Subject nghiêm ngặt với threshold 85%"""
        valid_paths = list(set(subject_features.keys()) & set(ai_features.keys()))
        if len(valid_paths) < 2:
            return []
        
        dsu = DSU(valid_paths)
        num_comparisons = len(valid_paths) * (len(valid_paths) - 1) // 2
        comparison_count = 0
        
        for i, path1 in enumerate(valid_paths):
            for j, path2 in enumerate(valid_paths[i+1:], i+1):
                self._check_pause()
                if not self.is_running: return []
                
                comparison_count += 1
                if comparison_count % 50 == 0:  # Giảm frequency update
                    self.progress_updated.emit(
                        comparison_count, num_comparisons,
                        f"Tầng 4/5: Phân tích nghiêm ngặt {comparison_count}/{num_comparisons}"
                    )
                
                # Kết hợp Subject similarity và AI similarity với trọng số mới
                subject_sim = self._calculate_subject_similarity(path1, path2)
                ai_sim = (ai_features[path1] @ ai_features[path2].T).item()
                
                # THAY ĐỔI TRỌNG SỐ: ưu tiên Subject analysis hơn
                combined_similarity = 0.75 * subject_sim + 0.25 * ai_sim
                
                # CHỈ GỘP KHI CẢ HAI ĐỀU CAO
                if combined_similarity >= self.similarity_threshold and subject_sim >= 0.80 and ai_sim >= 0.70:
                    dsu.union(path1, path2)

        # Tạo groups từ DSU
        groups = defaultdict(list)
        for path in valid_paths:
            root = dsu.find(path)
            groups[root].append(path)
        
        return [paths for paths in groups.values() if len(paths) > 1]

    def _validate_and_score_groups(self, subject_groups: List[List[str]], all_file_info: Dict[str, Dict]) -> List[Dict]:
        """Validation siêu nghiêm ngặt để loại bỏ nhóm không liên quan"""
        final_groups = []
        
        for i, group_paths in enumerate(subject_groups):
            self._check_pause()
            if not self.is_running: return []
            
            self.progress_updated.emit(i + 1, len(subject_groups), f"Tầng 5/5: Validation nghiêm ngặt {i+1}/{len(subject_groups)}")
            
            if len(group_paths) < 2:
                continue
            
            # VALIDATION CỰC KỲ NGHIÊM NGẶT
            subject_scores = []
            color_scores = []
            viewpoint_scores = []
            
            for j in range(len(group_paths)):
                for k in range(j + 1, len(group_paths)):
                    path1, path2 = group_paths[j], group_paths[k]
                    
                    # Lấy features để kiểm tra chi tiết
                    features1 = self._get_or_compute_subject_features(path1)
                    features2 = self._get_or_compute_subject_features(path2)
                    
                    if features1 and features2:
                        # Tính từng metric riêng biệt
                        color_sim = np.dot(features1['color'], features2['color']) / (
                            np.linalg.norm(features1['color']) * np.linalg.norm(features2['color']) + 1e-8
                        )
                        viewpoint_sim = np.dot(features1['viewpoint'], features2['viewpoint']) / (
                            np.linalg.norm(features1['viewpoint']) * np.linalg.norm(features2['viewpoint']) + 1e-8
                        )
                        
                        subject_sim = self._calculate_subject_similarity(path1, path2)
                        
                        subject_scores.append(subject_sim)
                        color_scores.append(color_sim)
                        viewpoint_scores.append(viewpoint_sim)
            
            if not subject_scores:
                continue
                
            # Tính điểm trung bình
            avg_subject_score = sum(subject_scores) / len(subject_scores)
            avg_color_score = sum(color_scores) / len(color_scores)
            avg_viewpoint_score = sum(viewpoint_scores) / len(viewpoint_scores)
            
            # NGƯỠNG VALIDATION CỰC KỲ CAO
            validation_passed = (
                avg_subject_score >= 0.80 and  # Subject tổng thể >= 80%
                avg_color_score >= 0.75 and    # Màu sắc >= 75% (quan trọng nhất)
                avg_viewpoint_score >= 0.65 and # Viewpoint >= 65%
                min(subject_scores) >= 0.70     # TẤT CẢ các cặp phải >= 70%
            )
            
            if validation_passed:
                final_groups.append({
                    "type": "similar_subject",
                    "files": [all_file_info[p] for p in group_paths if p in all_file_info],
                    "score": avg_subject_score,
                    "analysis_method": f"Subject Analysis V25+ (Color:{avg_color_score:.2f} View:{avg_viewpoint_score:.2f})",
                    "validation_level": "NGHIÊM NGẶT"
                })
        
        return final_groups

    def stop(self):
        self.is_running = False
        self.resume()
        # Cleanup thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)

    def pause(self):
        self.mutex.lock()
        self._is_paused = True
        self.mutex.unlock()

    def resume(self):
        self.mutex.lock()
        self._is_paused = False
        self.mutex.unlock()
        self.pause_condition.wakeAll()
