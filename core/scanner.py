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

    def __init__(self, file_list, similarity_threshold):
        super().__init__()
        self.file_list = file_list
        # TĂNG NGƯỠNG để chỉ gộp Subject thực sự giống nhau
        self.similarity_threshold = 0.85  # Tăng từ 70% lên 85% cho độ chính xác cao
        self.structural_similarity_threshold = 30  # Tăng ngưỡng cấu trúc
        self.is_running = True
        self.mutex = QMutex()
        self.pause_condition = QWaitCondition()
        self._is_paused = False
        self.model: Optional[CLIP] = None
        self.preprocess: Optional[Callable] = None
        self.tokenizer: Optional[Callable] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Tối ưu batch size cho GPU
        self.batch_size = 64 if self.device == "cuda" else 8
        self.orb_cache: Dict[str, Any] = {}
        # Cache cho 4 yếu tố phân tích Subject
        self.subject_analysis_cache: Dict[str, Dict[str, Any]] = {}

    def _load_model(self):
        self.progress_updated.emit(0, 0, "Đang tải AI Model chuyên sâu cho Subject Analysis...")
        try:
            # Tối ưu GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
            
            model_name = 'ViT-H-14'
            pretrained_tag = 'laion2b_s32b_b79k'
            model_obj, _, preprocess_fn = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained_tag,
                device=self.device,
                cache_dir=os.path.join("core", "models")
            )
            self.model = cast(CLIP, model_obj)
            self.preprocess = cast(Callable, preprocess_fn)
            self.tokenizer = open_clip.get_tokenizer(model_name)
            if self.model: 
                self.model.eval()
                # Tối ưu memory với half precision trên GPU
                if self.device == "cuda":
                    self.model = self.model.half()
            print(f"INFO: AI Model optimized for {self.device.upper()} - Subject Analysis Ready")
        except Exception as e:
            raise RuntimeError(f"Không thể tải AI Model cho Subject Analysis.\nLỗi: {e}")

    def _get_file_info(self, file_path):
        try:
            with Image.open(file_path) as img:
                img.load()
                return {"path": file_path, "size": os.path.getsize(file_path), "width": img.width, "height": img.height}
        except (UnidentifiedImageError, FileNotFoundError, PermissionError, OSError):
            return None

    def _analyze_subject_detail(self, file_path: str) -> float:
        """Phân tích mức độ chi tiết của Subject"""
        if file_path in self.subject_analysis_cache and 'detail' in self.subject_analysis_cache[file_path]:
            return self.subject_analysis_cache[file_path]['detail']
        
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return 0.0
            
            # Sử dụng Laplacian để đo độ sắc nét
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            # Canny edges để đếm chi tiết
            edges = cv2.Canny(img, 50, 150)
            edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
            
            # Kết hợp hai metrics
            detail_score = (laplacian_var / 1000.0 + edge_density * 10) / 2
            detail_score = min(1.0, detail_score)  # Normalize to 0-1
            
            if file_path not in self.subject_analysis_cache:
                self.subject_analysis_cache[file_path] = {}
            self.subject_analysis_cache[file_path]['detail'] = detail_score
            return detail_score
        except Exception:
            return 0.0

    def _analyze_subject_color(self, file_path: str) -> np.ndarray:
        """Phân tích phổ màu chủ đạo của Subject"""
        if file_path in self.subject_analysis_cache and 'color' in self.subject_analysis_cache[file_path]:
            return self.subject_analysis_cache[file_path]['color']
        
        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                # Resize để tăng tốc xử lý
                img = img.resize((128, 128), Image.Resampling.LANCZOS)
                
                # Tính histogram RGB
                r_hist = np.histogram(np.array(img)[:, :, 0], bins=8, range=(0, 256))[0]
                g_hist = np.histogram(np.array(img)[:, :, 1], bins=8, range=(0, 256))[0]
                b_hist = np.histogram(np.array(img)[:, :, 2], bins=8, range=(0, 256))[0]
                
                # Normalize
                total_pixels = img.size[0] * img.size[1]
                color_vector = np.concatenate([r_hist, g_hist, b_hist]) / total_pixels
                
                if file_path not in self.subject_analysis_cache:
                    self.subject_analysis_cache[file_path] = {}
                self.subject_analysis_cache[file_path]['color'] = color_vector
                return color_vector
        except Exception:
            return np.zeros(24)  # 8+8+8 bins

    def _analyze_subject_viewpoint(self, file_path: str) -> np.ndarray:
        """Phân tích góc nhìn và composition của Subject"""
        if file_path in self.subject_analysis_cache and 'viewpoint' in self.subject_analysis_cache[file_path]:
            return self.subject_analysis_cache[file_path]['viewpoint']
        
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return np.zeros(9)
            
            # Chia ảnh thành lưới 3x3 để phân tích composition
            h, w = img.shape
            grid_features = []
            for i in range(3):
                for j in range(3):
                    y1, y2 = i * h // 3, (i + 1) * h // 3
                    x1, x2 = j * w // 3, (j + 1) * w // 3
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
            return np.zeros(9)

    def _analyze_subject_material(self, file_path: str) -> np.ndarray:
        """Phân tích texture và material của Subject"""
        if file_path in self.subject_analysis_cache and 'material' in self.subject_analysis_cache[file_path]:
            return self.subject_analysis_cache[file_path]['material']
        
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return np.zeros(6)
            
            # Resize để tăng tốc
            img = cv2.resize(img, (128, 128))
            
            # LBP (Local Binary Pattern) để phân tích texture
            def calculate_lbp(image, radius=1, n_points=8):
                lbp = np.zeros_like(image)
                for i in range(radius, image.shape[0] - radius):
                    for j in range(radius, image.shape[1] - radius):
                        center = image[i, j]
                        binary_string = ""
                        for k in range(n_points):
                            angle = 2 * np.pi * k / n_points
                            x = int(i + radius * np.cos(angle))
                            y = int(j + radius * np.sin(angle))
                            if image[x, y] >= center:
                                binary_string += "1"
                            else:
                                binary_string += "0"
                        lbp[i, j] = int(binary_string, 2)
                return lbp
            
            # Simplified texture analysis
            # Gabor filters simulation với convolution
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            
            # Texture features
            features = [
                float(np.var(img.astype(np.float32))) / 10000.0,  # Variance
                float(np.mean(np.abs(sobel_x))) / 255.0,  # Horizontal edges
                float(np.mean(np.abs(sobel_y))) / 255.0,  # Vertical edges
                float(np.std(img.astype(np.float32))) / 255.0,  # Standard deviation
                len(np.unique(img)) / 256.0,  # Unique values ratio
                float(np.mean(cv2.Laplacian(img, cv2.CV_64F).astype(np.float32))) / 255.0  # Smoothness
            ]
            
            material_vector = np.array(features)
            
            if file_path not in self.subject_analysis_cache:
                self.subject_analysis_cache[file_path] = {}
            self.subject_analysis_cache[file_path]['material'] = material_vector
            return material_vector
        except Exception:
            return np.zeros(6)

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

            self._load_model()
            if not self.model or not self.preprocess or not self.tokenizer:
                self.error_occurred.emit("AI Model chưa được tải cho Subject Analysis."); return

            # Tầng 1: Lọc Trùng lặp Tuyệt đối bằng Perceptual Hash
            total_files = len(self.file_list)
            hashes = {}
            all_file_info = {}
            self.progress_updated.emit(0, total_files, f"Tầng 1/5: Phát hiện ảnh trùng lặp...")
            
            for idx, file_path in enumerate(self.file_list):
                self._check_pause();
                if not self.is_running: return
                self.progress_updated.emit(idx + 1, total_files, f"Tầng 1/5: Hash ảnh ({idx+1}/{total_files})")
                info = self._get_file_info(file_path)
                if not info: continue
                all_file_info[file_path] = info
                try:
                    with Image.open(file_path) as img:
                        h = imagehash.dhash(img.convert("RGB"))
                        hashes.setdefault(h, []).append(file_path)
                except Exception: continue
            
            duplicate_groups = []
            files_for_deep_scan = []
            for h, paths in hashes.items():
                if len(paths) > 1:
                    duplicate_groups.append({ 
                        "type": "duplicate", 
                        "files": [all_file_info[p] for p in paths if p in all_file_info], 
                        "score": 1.0 
                    })
                elif paths:
                    files_for_deep_scan.append(paths[0])
            
            # Tầng 2: Phân tích Subject với 4 yếu tố (Detail + Color + Viewpoint + Material)
            num_deep_scan = len(files_for_deep_scan)
            self.progress_updated.emit(0, num_deep_scan, "Tầng 2/5: Phân tích Subject (4 yếu tố)...")
            
            subject_features = {}
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
        """Trích xuất AI features với batch processing tối ưu"""
        ai_features = {}
        
        for i in range(0, len(file_paths), self.batch_size):
            self._check_pause();
            if not self.is_running: return {}
            
            batch_paths = file_paths[i:i+self.batch_size]
            batch_images, valid_paths_in_batch = [], []
            
            for file_path in batch_paths:
                try:
                    with Image.open(file_path) as img:
                        # Chuyển đổi precision cho GPU
                        if self.preprocess is not None:
                            image_input = self.preprocess(img).to(self.device)
                            if self.device == "cuda":
                                image_input = image_input.half()
                            batch_images.append(image_input)
                            valid_paths_in_batch.append(file_path)
                except Exception: 
                    continue
            
            if not batch_images: continue
            
            self.progress_updated.emit(
                i + len(batch_paths), total_files, 
                f"Tầng 3/5: AI Features lô {(i//self.batch_size)+1}... ({i+len(batch_paths)}/{total_files})"
            )
            
            batch_tensor = torch.stack(batch_images)
            if self.model is not None:
                image_features = self.model.encode_image(batch_tensor)
                
                if image_features is not None:
                    image_features = image_features.float()  # Convert back to float32
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    for path, feature_vec in zip(valid_paths_in_batch, image_features):
                        ai_features[path] = feature_vec.unsqueeze(0)
        
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

    def pause(self):
        self.mutex.lock()
        self._is_paused = True
        self.mutex.unlock()

    def resume(self):
        self.mutex.lock()
        self._is_paused = False
        self.mutex.unlock()
        self.pause_condition.wakeAll()
