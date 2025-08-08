# core/scanner.py
#
# Phiên bản V29: Tối ưu hóa hiệu suất.
# - [REMOVED] Loại bỏ hoàn toàn logic tải model khỏi Worker.
# - [CHANGED] Worker giờ đây nhận một model đã được tải sẵn khi khởi tạo,
#   giúp giảm độ trễ khi bắt đầu quét và tách biệt logic rõ ràng hơn.

import os
import numpy as np
from typing import Optional, Callable, cast, List, Dict, Any, Tuple, Tuple
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition, QMutexLocker
from PIL import Image, UnidentifiedImageError
import imagehash
import torch
from torch import Tensor
import open_clip
from open_clip import CLIP
import cv2
from collections import defaultdict
import gc

class DSU:
    def __init__(self, items):
        self.parent = {item: item for item in items}
    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j: self.parent[root_j] = root_i

class ScannerWorker(QThread):
    progress_updated = Signal(int, int, str)
    scan_completed = Signal(list)
    error_occurred = Signal(str)

    # [CHANGED] __init__ giờ đây nhận model, preprocess, và tokenizer đã được tải sẵn
    def __init__(self, file_list, similarity_threshold, model: CLIP, preprocess: Callable, tokenizer: Callable, speed_mode='high_quality'):
        super().__init__()
        self.file_list = file_list
        from speed_config import SpeedConfig
        self.speed_config = SpeedConfig.get_mode(speed_mode)
        self.similarity_threshold = self.speed_config['similarity_threshold']
        
        # Nhận các đối tượng đã được tải sẵn
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        
        # Add timing for progress calculation
        self.start_time = None
        
        # Metadata for hybrid group detection
        self.group_metadata = {}
        
        self.is_running = True
        self.mutex = QMutex()
        self.pause_condition = QWaitCondition()
        self._is_paused = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = self.speed_config['batch_size_gpu'] if self.device == "cuda" else self.speed_config['batch_size_cpu']
        self.orb_cache: Dict[str, Any] = {}
        self.subject_analysis_cache: Dict[str, Dict[str, Any]] = {}

    # [REMOVED] Hàm _load_model() đã được loại bỏ hoàn toàn.
    
    def _get_file_info(self, file_path):
        try:
            with Image.open(file_path) as img:
                img.load()
                # Calculate dhash for duplicate detection
                dhash = imagehash.dhash(img.convert("RGB"))
                return {
                    "path": file_path, 
                    "size": os.path.getsize(file_path), 
                    "width": img.width, 
                    "height": img.height,
                    "dhash": dhash
                }
        except (UnidentifiedImageError, FileNotFoundError, PermissionError, OSError):
            return None

    def _analyze_subject_detail(self, file_path: str) -> float:
        if file_path in self.subject_analysis_cache and 'detail' in self.subject_analysis_cache[file_path]: return self.subject_analysis_cache[file_path]['detail']
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return 0.0
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            detail_score = min(1.0, laplacian_var / 1000.0)
            if file_path not in self.subject_analysis_cache: self.subject_analysis_cache[file_path] = {}
            self.subject_analysis_cache[file_path]['detail'] = detail_score
            return detail_score
        except Exception: return 0.0

    def _analyze_subject_color(self, file_path: str) -> np.ndarray:
        if file_path in self.subject_analysis_cache and 'color' in self.subject_analysis_cache[file_path]: return self.subject_analysis_cache[file_path]['color']
        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB').resize((128, 128), Image.Resampling.LANCZOS)
                bins = 8
                img_array = np.array(img)
                r_hist, g_hist, b_hist = (np.histogram(img_array[:, :, i], bins=bins, range=(0, 256))[0] for i in range(3))
                color_vector = np.concatenate([r_hist, g_hist, b_hist]) / (img.size[0] * img.size[1])
                if file_path not in self.subject_analysis_cache: self.subject_analysis_cache[file_path] = {}
                self.subject_analysis_cache[file_path]['color'] = color_vector
                return color_vector
        except Exception: return np.zeros(8 * 3)

    def _analyze_subject_viewpoint(self, file_path: str) -> np.ndarray:
        if file_path in self.subject_analysis_cache and 'viewpoint' in self.subject_analysis_cache[file_path]: return self.subject_analysis_cache[file_path]['viewpoint']
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return np.zeros(3 * 3)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            grid_size = 3
            h, w = img.shape
            grid_features = [float(np.mean(img[i*h//grid_size:(i+1)*h//grid_size, j*w//grid_size:(j+1)*w//grid_size].astype(float)))/255.0 for i in range(grid_size) for j in range(grid_size)]
            viewpoint_vector = np.array(grid_features)
            if file_path not in self.subject_analysis_cache: self.subject_analysis_cache[file_path] = {}
            self.subject_analysis_cache[file_path]['viewpoint'] = viewpoint_vector
            return viewpoint_vector
        except Exception: return np.zeros(3 * 3)

    def _analyze_subject_material(self, file_path: str) -> np.ndarray:
        if file_path in self.subject_analysis_cache and 'material' in self.subject_analysis_cache[file_path]: return self.subject_analysis_cache[file_path]['material']
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return np.zeros(6)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            features = [
                float(np.var(img.astype(float)))/10000.0, float(np.mean(np.abs(sobel_x).astype(float)))/255.0,
                float(np.mean(np.abs(sobel_y).astype(float)))/255.0, float(np.std(img.astype(float)))/255.0,
                len(np.unique(img))/256.0, float(np.mean(cv2.Laplacian(img, cv2.CV_64F).astype(float)))/255.0
            ]
            material_vector = np.array(features)
            if file_path not in self.subject_analysis_cache: self.subject_analysis_cache[file_path] = {}
            self.subject_analysis_cache[file_path]['material'] = material_vector
            return material_vector
        except Exception: return np.zeros(6)

    def _get_or_compute_subject_features(self, file_path: str) -> Optional[Dict[str, Any]]:
        if file_path in self.orb_cache: return self.orb_cache[file_path]
        try:
            subject_features = {
                'detail': self._analyze_subject_detail(file_path),
                'color': self._analyze_subject_color(file_path),
                'viewpoint': self._analyze_subject_viewpoint(file_path),
                'material': self._analyze_subject_material(file_path)
            }
            self.orb_cache[file_path] = subject_features
            return subject_features
        except Exception: return None

    def _calculate_subject_similarity(self, path1: str, path2: str) -> float:
        """
        Tính điểm tương tự CHÍNH XÁC tập trung vào CHỦ THỂ và MÀU SẮC:
        - Màu sắc: 60% (yếu tố chính để tránh gộp nhầm)
        - Chi tiết chủ thể: 30% (texture, pattern của chủ thể)
        - Góc nhìn: 10% (composition tương tự)
        """
        features1 = self._get_or_compute_subject_features(path1)
        features2 = self._get_or_compute_subject_features(path2)
        if features1 is None or features2 is None: return 0.0
        
        try:
            # Trọng số mới tập trung CHỦ THỂ và MÀU SẮC để tránh gộp nhầm
            weights = {
                'color': 0.60,      # MÀU SẮC - yếu tố quan trọng nhất (tăng từ 50% → 60%)
                'detail': 0.30,     # CHI TIẾT CHỦ THỂ - texture và pattern (tăng từ 25% → 30%)
                'viewpoint': 0.10,  # GÓC NHÌN - composition (giảm từ 15% → 10%)
                # Loại bỏ material để tập trung hơn
            }
            
            # Tính toán similarity CHẶT CHẼ cho từng thành phần
            sims = {}
            
            # Color similarity với THRESHOLD CAO để tránh gộp nhầm
            color_norm1 = np.linalg.norm(features1['color'])
            color_norm2 = np.linalg.norm(features2['color'])
            if color_norm1 > 1e-8 and color_norm2 > 1e-8:
                color_sim = np.dot(features1['color'], features2['color']) / (color_norm1 * color_norm2)
                # Áp dụng penalty mạnh nếu màu sắc khác biệt
                if color_sim < 0.85:  # Nếu màu sắc không đủ tương tự
                    color_sim = color_sim * 0.5  # Penalty mạnh
                sims['color'] = max(0.0, color_sim)
            else:
                sims['color'] = 0.0
            
            # Detail similarity với threshold cao cho CHỦ THỂ
            detail_sim = 1.0 - abs(features1['detail'] - features2['detail'])
            if detail_sim < 0.80:  # Chi tiết chủ thể không đủ tương tự
                detail_sim = detail_sim * 0.6  # Penalty
            sims['detail'] = max(0.0, detail_sim)
                
            # Viewpoint similarity - ít quan trọng hơn
            view_norm1 = np.linalg.norm(features1['viewpoint'])
            view_norm2 = np.linalg.norm(features2['viewpoint'])
            if view_norm1 > 1e-8 and view_norm2 > 1e-8:
                sims['viewpoint'] = np.dot(features1['viewpoint'], features2['viewpoint']) / (view_norm1 * view_norm2)
            else:
                sims['viewpoint'] = 0.0
            # Tính tổng điểm với trọng số MỚI (chỉ 3 yếu tố)
            total_similarity = sum(weights[k] * max(0.0, sims[k]) for k in weights)
            
            # ANTI-GROUPING MISTAKE PENALTY SYSTEM
            penalties = 0.0
            
            # PENALTY MẠH nếu màu sắc quá khác biệt (quan trọng nhất)
            if sims['color'] < 0.70:
                penalties += 0.40  # Penalty rất mạnh cho màu sắc khác biệt
            elif sims['color'] < 0.85:
                penalties += 0.20  # Penalty mạnh cho màu sắc hơi khác
                
            # PENALTY CHO chi tiết chủ thể khác biệt
            if sims['detail'] < 0.65:
                penalties += 0.30  # Penalty mạnh cho chi tiết chủ thể khác
            elif sims['detail'] < 0.80:
                penalties += 0.15  # Penalty nhẹ
                
            # PENALTY NHẸ cho góc nhìn (ít quan trọng)
            if sims['viewpoint'] < 0.40:
                penalties += 0.10
            
            # EXTRA PENALTY nếu cả màu sắc VÀ chi tiết đều kém
            if sims['color'] < 0.75 and sims['detail'] < 0.75:
                penalties += 0.25  # Double penalty để tránh gộp nhầm
            
            # Áp dụng penalty và normalize về [0,1]
            final_score = max(0.0, min(1.0, total_similarity - penalties))
            
            return final_score
            
        except Exception as e:
            return 0.0

    def _check_pause(self):
        with QMutexLocker(self.mutex):
            while self._is_paused:
                self.pause_condition.wait(self.mutex)

    def _prescan_and_fast_hash(self, file_list: List[str]) -> Tuple[Dict, Dict]:
        """
        Tầng 0: Pre-scan nhanh để tối ưu cho 10K+ files
        Returns: (all_file_info, fast_duplicates)
        """
        all_file_info = {}
        hash_to_files = defaultdict(list)
        fast_duplicates = defaultdict(list)
        
        total_files = len(file_list)
        for idx, file_path in enumerate(file_list):
            self._check_pause()
            if not self.is_running: return {}, {}
            
            # Progress update every 100 files for speed
            if idx % 100 == 0 or idx == total_files - 1:
                percentage = ((idx + 1) / total_files) * 100
                self.progress_updated.emit(
                    idx + 1, 
                    total_files, 
                    f"Tầng 0/7: Pre-scan ({idx+1}/{total_files} - {percentage:.1f}%)"
                )
            
            file_info = self._get_file_info(file_path)
            if file_info:
                all_file_info[file_path] = file_info
                dhash = file_info.get('dhash')
                if dhash:
                    hash_to_files[dhash].append(file_path)
        
        # Identify fast duplicates (exact hash matches)
        for dhash, paths in hash_to_files.items():
            if len(paths) > 1:
                fast_duplicates[dhash] = paths
                
        return all_file_info, dict(fast_duplicates)

    def _build_similarity_matrix(self, valid_files: List[str], ai_features: Dict[str, Tensor]) -> Dict:
        """
        Tầng 3: Xây dựng similarity matrix để tối ưu clustering
        """
        similarity_matrix = {}
        total_pairs = len(valid_files) * (len(valid_files) - 1) // 2
        processed_pairs = 0
        
        for i, path1 in enumerate(valid_files):
            for j, path2 in enumerate(valid_files[i+1:], i+1):
                self._check_pause()
                if not self.is_running: return {}
                
                processed_pairs += 1
                if processed_pairs % 500 == 0 or processed_pairs == total_pairs:
                    percentage = (processed_pairs / total_pairs) * 100
                    self.progress_updated.emit(
                        processed_pairs,
                        total_pairs,
                        f"Tầng 3/7: Building Matrix ({processed_pairs}/{total_pairs} - {percentage:.1f}%)"
                    )
                
                # Calculate similarities
                subject_sim = self._calculate_subject_similarity(path1, path2)
                
                feature1 = ai_features[path1].to(self.device)
                feature2 = ai_features[path2].to(self.device)
                ai_sim = torch.dot(feature1, feature2).item()
                
                # Store in matrix
                key = (path1, path2) if path1 < path2 else (path2, path1)
                similarity_matrix[key] = {
                    'subject_sim': subject_sim,
                    'ai_sim': ai_sim,
                    'combined_sim': 0.65 * subject_sim + 0.35 * ai_sim  # Balanced for hybrid detection
                }
                
        return similarity_matrix

    def _perform_advanced_clustering(self, valid_files: List[str], similarity_matrix: Dict, 
                                   fast_duplicates: Dict, all_file_info: Dict) -> List[List[str]]:
        """
        Tầng 4: Advanced clustering với hybrid detection chuẩn
        """
        dsu = DSU(valid_files)
        
        # Step 1: Union all fast duplicates first
        for dhash, duplicate_paths in fast_duplicates.items():
            for i in range(1, len(duplicate_paths)):
                dsu.union(duplicate_paths[0], duplicate_paths[i])
        
        # Step 2: Process similarity matrix for similar groups
        processed_pairs = 0
        total_pairs = len(similarity_matrix)
        
        for (path1, path2), sim_data in similarity_matrix.items():
            self._check_pause()
            if not self.is_running: return []
            
            processed_pairs += 1
            if processed_pairs % 200 == 0 or processed_pairs == total_pairs:
                percentage = (processed_pairs / total_pairs) * 100
                self.progress_updated.emit(
                    processed_pairs,
                    total_pairs,
                    f"Tầng 4/7: Clustering ({processed_pairs}/{total_pairs} - {percentage:.1f}%)"
                )
            
            combined_sim = sim_data['combined_sim']
            subject_sim = sim_data['subject_sim']
            ai_sim = sim_data['ai_sim']
            
            # Advanced thresholds for 10K+ files accuracy
            if (combined_sim >= self.similarity_threshold and 
                subject_sim >= 0.72 and  # Slightly lower for hybrid capture
                ai_sim >= 0.68):         # Slightly lower for hybrid capture
                
                dsu.union(path1, path2)
        
        # Step 3: Extract groups and analyze composition
        groups = defaultdict(list)
        for path in valid_files:
            groups[dsu.find(path)].append(path)
        
        final_groups = []
        for group_paths in groups.values():
            if len(group_paths) > 1:
                # Analyze group composition for hybrid detection
                group_analysis = self._analyze_group_composition(group_paths, fast_duplicates, all_file_info)
                final_groups.append(group_paths)
                
                # Store metadata for finalize phase
                group_key = tuple(sorted(group_paths))
                self.group_metadata[group_key] = group_analysis
        
        return final_groups

    def _analyze_group_composition(self, group_paths: List[str], fast_duplicates: Dict, all_file_info: Dict) -> Dict:
        """
        Phân tích composition của group để xác định hybrid chính xác
        """
        duplicate_files = set()
        similar_files = set(group_paths)
        
        # Find which files are duplicates within this group
        for dhash, dup_paths in fast_duplicates.items():
            group_duplicates = [p for p in dup_paths if p in group_paths]
            if len(group_duplicates) > 1:
                duplicate_files.update(group_duplicates)
        
        # Files that are similar but not duplicates
        similar_only_files = similar_files - duplicate_files
        
        return {
            'total_files': len(group_paths),
            'duplicate_files': len(duplicate_files),
            'similar_only_files': len(similar_only_files),
            'has_duplicates': len(duplicate_files) > 0,
            'has_similars': len(similar_only_files) > 0,
            'is_hybrid': len(duplicate_files) > 0 and len(similar_only_files) > 0,
            'duplicate_paths': list(duplicate_files),
            'similar_only_paths': list(similar_only_files),
            'composition_type': self._determine_composition_type(len(duplicate_files), len(similar_only_files))
        }
    
    def _determine_composition_type(self, dup_count: int, sim_count: int) -> str:
        """Xác định loại composition chính xác"""
        if dup_count == 0:
            return "pure_similar"
        elif sim_count == 0:
            return "pure_duplicate" 
        else:
            return "hybrid"

    def run(self):
        """
        Vòng lặp chính với kiến trúc 7 tầng quét siêu chuẩn cho 10K+ Files
        
        Tầng 1: ⭐ DUPLICATE DETECTION PRIORITY - Quét trùng lặp ưu tiên
        Tầng 2: AI Feature Extraction & Matrix Building  
        Tầng 3: Advanced DSU Clustering
        Tầng 4: Group Composition Analysis
        Tầng 5: Cross-Validation & Recovery
        Tầng 6: Ultra-Strict Quality Control (0.82-0.85)
        Tầng 7: Final Validation & Cross-Optimization
        """
        try:
            import time
            self.start_time = time.time()  # Record start time
            
            if len(self.file_list) < 2: 
                self.scan_completed.emit([])
                return

            # Tối ưu thông số cho dataset lớn
            self._optimize_for_large_dataset(len(self.file_list))

            # Logic tải model đã được chuyển ra ngoài
            if not self.model or not self.preprocess:
                self.error_occurred.emit("Lỗi: Model AI chưa được tải.")
                return

            # Tầng 1: 🏆 DUPLICATE DETECTION PRIORITY (15-20% thời gian)
            self.progress_updated.emit(0, 100, "Tầng 1/7: 🏆 DUPLICATE DETECTION PRIORITY...")
            all_file_info, comprehensive_duplicates = self._comprehensive_duplicate_detection(self.file_list)
            valid_files = list(all_file_info.keys())
            if not self.is_running or not all_file_info:
                self.scan_completed.emit([])
                return

            # Tầng 2: AI Feature Extraction & Matrix Building (20-25% thời gian)
            self.progress_updated.emit(20, 100, "Tầng 2/7: AI Feature Extraction & Matrix Building...")
            # Trích xuất AI features trước
            ai_features = self._extract_ai_features_batch(valid_files)
            similarity_matrix = self._build_similarity_matrix(valid_files, ai_features)
            if not self.is_running or similarity_matrix is None:
                self.scan_completed.emit([])
                return

            # Tầng 3: Advanced DSU Clustering (15-18% thời gian)
            self.progress_updated.emit(45, 100, "Tầng 3/7: Advanced DSU Clustering...")
            clustered_groups = self._perform_advanced_clustering(valid_files, similarity_matrix, comprehensive_duplicates, all_file_info)
            if not self.is_running or not clustered_groups:
                self.scan_completed.emit([])
                return

            # Tầng 4: Group Composition Analysis (8-12% thời gian) 
            self.progress_updated.emit(63, 100, "Tầng 4/7: Group Composition Analysis...")
            # Phân tích composition cho từng nhóm
            for group_paths in clustered_groups:
                if len(group_paths) >= 2:
                    self._analyze_group_composition(group_paths, all_file_info, all_file_info)
            
            if not self.is_running:
                self.scan_completed.emit([])
                return

            # Tầng 5: Cross-Validation & Recovery (8-10% thời gian)
            self.progress_updated.emit(75, 100, "Tầng 5/7: Cross-Validation & Recovery...")
            cross_validated_groups = self._cross_validate_groups(clustered_groups, all_file_info, similarity_matrix)
            if not self.is_running:
                self.scan_completed.emit([])
                return

            # Tầng 6: ⚡ Anti-Mistake Quality Control (0.88-0.92) (8-10% thời gian)
            self.progress_updated.emit(85, 100, "Tầng 6/7: ⚡ Anti-Mistake Quality Control (0.88-0.92)...")
            quality_controlled_groups = self._ultra_strict_quality_control(cross_validated_groups, all_file_info, similarity_matrix)
            
            if not self.is_running:
                self.scan_completed.emit([])
                return

            # Tầng 7: Final Validation & Cross-Optimization (6-8% thời gian)
            self.progress_updated.emit(93, 100, "Tầng 7/7: Final Validation & Cross-Optimization...")
            final_groups = self._final_validation_and_cross_optimize(quality_controlled_groups, all_file_info)

            # Tầng 8: Enhanced Rescan Validation để phát hiện file bị bỏ sót
            self.progress_updated.emit(97, 100, "Tầng 8/8: Enhanced Rescan - Phát hiện file bị bỏ sót...")
            final_groups = self._enhanced_rescan_validation(final_groups, all_file_info)

            # Sắp xếp và hoàn thiện kết quả
            sort_order = {"duplicate": 0, "hybrid_subject": 1, "similar_subject": 2}
            final_groups.sort(key=lambda g: (sort_order.get(g.get('type', ''), 3), -g.get('score', 0.0)))
            
            if self.device == "cuda": torch.cuda.empty_cache()
            gc.collect()
            
            # Emit kết quả cuối cùng
            self.progress_updated.emit(100, 100, "✅ Hoàn thành 8 tầng với Enhanced Rescan!")
            self.scan_completed.emit(final_groups)
            
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Lỗi quét:\n{e}\n{traceback.format_exc()}")

    def _extract_ai_features_batch(self, file_paths: List[str]) -> Dict[str, Tensor]:
        ai_features = {}
        total_files = len(file_paths)
        processed_files = 0
        
        for i in range(0, total_files, self.batch_size):
            self._check_pause()
            if not self.is_running: return {}
            
            batch_paths = file_paths[i:i+self.batch_size]
            batch_images, valid_paths = [], []
            
            # Load batch with individual file progress
            for j, path in enumerate(batch_paths):
                try:
                    with Image.open(path) as img:
                        if self.preprocess and callable(self.preprocess):
                            image_input = self.preprocess(img).to(self.device)
                            if self.device == "cuda": image_input = image_input.half()
                            batch_images.append(image_input)
                            valid_paths.append(path)
                            
                            # Update progress per file for smoother experience
                            processed_files += 1
                            if processed_files % 5 == 0 or processed_files == total_files:
                                percentage = (processed_files / total_files) * 100
                                self.progress_updated.emit(
                                    processed_files, 
                                    total_files, 
                                    f"Tầng 2/7: Phân tích AI ({processed_files}/{total_files} - {percentage:.1f}%)"
                                )
                except Exception: 
                    processed_files += 1
                    continue
            
            if not batch_images: 
                continue

            # Process batch
            with torch.no_grad():
                if self.model and callable(self.model.encode_image):
                    image_features = cast(Optional[Tensor], self.model.encode_image(torch.stack(batch_images)))
                    
                    if image_features is not None:
                        normalized_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        
                        for path, vec in zip(valid_paths, normalized_features):
                            ai_features[path] = vec.cpu()
                        
                        del image_features
                        del normalized_features

            del batch_images
            if self.device == "cuda": 
                torch.cuda.empty_cache()
                
        return ai_features

    def _perform_ai_subject_clustering(self, valid_paths: List[str], ai_features: Dict[str, Tensor], all_file_info: Dict) -> List[List[str]]:
        if len(valid_paths) < 2: return []
        dsu = DSU(valid_paths)
        num_comparisons = len(valid_paths) * (len(valid_paths) - 1) // 2
        comparison_count = 0
        
        # Track duplicate pairs for hybrid group detection
        duplicate_pairs = set()
        similar_pairs = set()
        
        # Improved progress tracking
        progress_interval = max(1, num_comparisons // 100)  # Update every 1% or at least every comparison
        
        for i, path1 in enumerate(valid_paths):
            for j, path2 in enumerate(valid_paths[i+1:], i+1):
                self._check_pause()
                if not self.is_running: return []
                
                comparison_count += 1
                # More frequent and smooth progress updates
                if comparison_count % progress_interval == 0 or comparison_count == num_comparisons:
                    percentage = (comparison_count / num_comparisons) * 100
                    self.progress_updated.emit(
                        comparison_count, 
                        num_comparisons, 
                        f"Tầng 3/7: So sánh cặp {comparison_count}/{num_comparisons} ({percentage:.1f}%)"
                    )
                
                is_duplicate = False
                
                # Check for exact duplicates first using hash
                file1_info = all_file_info.get(path1)
                file2_info = all_file_info.get(path2)
                
                if file1_info and file2_info:
                    hash1 = file1_info.get('dhash')
                    hash2 = file2_info.get('dhash')
                    
                    # More lenient hash comparison for duplicates
                    if hash1 and hash2:
                        hash_diff = hash1 - hash2  # Hamming distance
                        if hash_diff <= 5:  # More lenient threshold for duplicates
                            dsu.union(path1, path2)
                            duplicate_pairs.add((path1, path2))
                            is_duplicate = True
                            continue  # Skip expensive similarity calculation
                
                # Calculate detailed similarity for non-duplicates or unclear cases
                if not is_duplicate:
                    subject_sim = self._calculate_subject_similarity(path1, path2)
                    
                    # AI similarity calculation
                    feature1 = ai_features[path1].to(self.device)
                    feature2 = ai_features[path2].to(self.device)
                    ai_sim = torch.dot(feature1, feature2).item()
                    
                    # Combined similarity with adjusted weights
                    combined_sim = 0.70 * subject_sim + 0.30 * ai_sim  # Give more weight to subject analysis
                    
                    # More flexible thresholds to capture hybrid groups
                    if combined_sim >= self.similarity_threshold and subject_sim >= 0.70 and ai_sim >= 0.65:
                        dsu.union(path1, path2)
                        similar_pairs.add((path1, path2))
                    
                # Memory cleanup for large datasets
                if comparison_count % 1000 == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()
                    
        groups = defaultdict(list)
        for path in valid_paths: groups[dsu.find(path)].append(path)
        
        # Store metadata about group composition for hybrid detection
        grouped_paths = [paths for paths in groups.values() if len(paths) > 1]
        for group in grouped_paths:
            # Check if this group contains both duplicates and similars
            has_duplicates = any((p1, p2) in duplicate_pairs for p1 in group for p2 in group if p1 != p2)
            has_similars = any((p1, p2) in similar_pairs for p1 in group for p2 in group if p1 != p2)
            
            # Mark hybrid groups in metadata (store in a way accessible to finalize)
            if not hasattr(self, 'group_metadata'):
                self.group_metadata = {}
            
            group_key = tuple(sorted(group))
            self.group_metadata[group_key] = {
                'has_duplicates': has_duplicates,
                'has_similars': has_similars,
                'is_hybrid': has_duplicates and has_similars
            }
        
        return grouped_paths

    def _classify_and_finalize_groups(self, initial_groups: List[List[str]], all_file_info: Dict) -> List[Dict]:
        final_groups = []
        total_groups = len(initial_groups)
        
        for i, group_paths in enumerate(initial_groups):
            # Smooth progress updates
            percentage = ((i + 1) / total_groups) * 100
            self.progress_updated.emit(
                i + 1, 
                total_groups, 
                f"Tầng 5/7: Phân loại nhóm {i+1}/{total_groups} ({percentage:.1f}%)"
            )
            
            if len(group_paths) < 2: continue
            
            # Pause check for responsiveness
            self._check_pause()
            if not self.is_running: return []
            
            # Get metadata about this group from advanced clustering
            group_key = tuple(sorted(group_paths))
            metadata = self.group_metadata.get(group_key, {})
            
            # Use metadata to determine accurate group type
            composition_type = metadata.get('composition_type', 'unknown')
            
            if composition_type == "pure_duplicate":
                group_type = "duplicate"
            elif composition_type == "hybrid":
                group_type = "hybrid_subject"
            elif composition_type == "pure_similar":
                group_type = "similar_subject"
            else:
                # Fallback to hash analysis if metadata missing
                hashes = defaultdict(list)
                for path in group_paths:
                    file_info = all_file_info.get(path)
                    if file_info and 'dhash' in file_info:
                        hashes[file_info['dhash']].append(path)
                
                num_duplicates = sum(1 for paths in hashes.values() if len(paths) > 1)
                num_unique_in_group = len(hashes)
                
                if num_unique_in_group == 1:
                    group_type = "duplicate"
                elif num_duplicates > 0:
                    group_type = "hybrid_subject"
                else:
                    group_type = "similar_subject"
            
            # Calculate average similarity score with the improved algorithm
            scores = []
            for j in range(len(group_paths)):
                for k in range(j + 1, len(group_paths)):
                    score = self._calculate_subject_similarity(group_paths[j], group_paths[k])
                    scores.append(score)
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            # Quality control optimized for 10K+ files
            quality_thresholds = {
                "duplicate": 0.0,       # Duplicates always pass
                "hybrid_subject": 0.65, # Lenient for hybrid to preserve valuable groups
                "similar_subject": 0.72  # Strict for pure similar
            }
            
            min_score = quality_thresholds.get(group_type, 0.70)
            if group_type != "duplicate" and avg_score < min_score:
                continue  # Skip low-quality groups
            
            # Enhanced file info with detailed metadata
            enhanced_files = []
            duplicate_paths = metadata.get('duplicate_paths', [])
            similar_paths = metadata.get('similar_only_paths', [])
            
            for path in group_paths:
                if path in all_file_info:
                    file_data = all_file_info[path].copy()
                    # Add hybrid-specific metadata
                    if 'dhash' in file_data:
                        file_data['hash_str'] = str(file_data['dhash'])
                    
                    # Mark file role in hybrid group
                    if path in duplicate_paths:
                        file_data['role_in_group'] = 'duplicate'
                        file_data['action_suggestion'] = 'delete_candidate'
                    elif path in similar_paths:
                        file_data['role_in_group'] = 'similar'
                        file_data['action_suggestion'] = 'rename_candidate'
                    else:
                        file_data['role_in_group'] = 'primary'
                        file_data['action_suggestion'] = 'keep'
                    
                    enhanced_files.append(file_data)
            
            # Enhanced group info with hybrid action plan
            group_info = {
                "type": group_type,
                "files": enhanced_files,
                "score": 1.0 if group_type == "duplicate" else avg_score,
                "analysis_method": f"V31+: {group_type.replace('_', ' ').title()} (10K+ Optimized)",
                "composition": metadata,
                "action_plan": self._generate_action_plan(group_type, metadata, enhanced_files)
            }
            
            final_groups.append(group_info)
            
        return final_groups

    def _generate_action_plan(self, group_type: str, metadata: Dict, files: List[Dict]) -> Dict:
        """
        Tạo action plan cụ thể cho từng loại group
        """
        if group_type == "duplicate":
            return {
                "strategy": "keep_one_delete_rest",
                "keep_file": "best_quality",  # Keep best quality file
                "delete_count": len(files) - 1,
                "action_sequence": ["delete_duplicates"]
            }
        elif group_type == "hybrid_subject":
            duplicate_count = metadata.get('duplicate_files', 0)
            similar_count = metadata.get('similar_only_files', 0)
            remaining_after_delete = len(files) - (duplicate_count - 1 if duplicate_count > 1 else 0)
            
            return {
                "strategy": "hybrid_two_phase", 
                "phase_1_delete_duplicates": duplicate_count - 1 if duplicate_count > 1 else 0,
                "phase_2_rename_remaining": remaining_after_delete,  # Đổi tên TẤT CẢ files còn lại
                "total_actions": (duplicate_count - 1 if duplicate_count > 1 else 0) + remaining_after_delete,
                "action_sequence": ["delete_duplicates", "rename_all_remaining"],
                "explanation": f"Xóa {duplicate_count - 1 if duplicate_count > 1 else 0} file trùng lặp, sau đó đổi tên {remaining_after_delete} file còn lại"
            }
        else:  # similar_subject
            return {
                "strategy": "rename_all_but_one",
                "keep_file": "best_quality",  # Keep highest quality
                "rename_count": len(files) - 1,
                "action_sequence": ["rename_similars"]
            }

    def _cross_validate_groups(self, groups: List[List[str]], all_file_info: Dict, similarity_matrix: Optional[Dict]) -> List[List[str]]:
        """
        Tầng 4: Cross-validation và re-check để đảm bảo không bỏ sót
        """
        validated_groups = []
        missed_files = set()
        
        # Collect all files already in groups
        grouped_files = set()
        for group in groups:
            grouped_files.update(group)
        
        # Find files that might have been missed
        all_files = set(all_file_info.keys())
        potential_missed = all_files - grouped_files
        
        self.progress_updated.emit(0, len(groups) + len(potential_missed), "Tầng 4/7: Cross-validating groups...")
        
        # Re-validate existing groups with stricter criteria
        for i, group in enumerate(groups):
            self._check_pause()
            if not self.is_running: return []
            
            if len(group) < 2:
                continue
                
            # Double-check group cohesion
            cohesion_scores = []
            for j in range(len(group)):
                for k in range(j + 1, len(group)):
                    score = self._calculate_subject_similarity(group[j], group[k])
                    cohesion_scores.append(score)
            
            avg_cohesion = sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0.0
            
            # More strict validation threshold
            if avg_cohesion >= 0.55:  # Lowered threshold to catch more groups
                validated_groups.append(group)
            else:
                # Break up weak groups and add files back to potential missed
                potential_missed.update(group)
            
            self.progress_updated.emit(i + 1, len(groups) + len(potential_missed), f"Tầng 4/7: Validating group {i+1}/{len(groups)}")
        
        # Re-check missed files for potential new groups
        missed_files_list = list(potential_missed)
        if len(missed_files_list) >= 2:
            # Quick re-clustering for missed files
            for i in range(len(missed_files_list)):
                self._check_pause()
                if not self.is_running: return validated_groups
                
                current_file = missed_files_list[i]
                if current_file in missed_files:  # Still unprocessed
                    new_group = [current_file]
                    missed_files.discard(current_file)
                    
                    # Find similar files
                    for j in range(i + 1, len(missed_files_list)):
                        other_file = missed_files_list[j]
                        if other_file in missed_files:
                            similarity = self._calculate_subject_similarity(current_file, other_file)
                            if similarity >= 0.60:  # Lenient threshold for recovery
                                new_group.append(other_file)
                                missed_files.discard(other_file)
                    
                    if len(new_group) >= 2:
                        validated_groups.append(new_group)
                
                progress_idx = len(groups) + i + 1
                self.progress_updated.emit(progress_idx, len(groups) + len(potential_missed), f"Tầng 4/7: Re-checking missed file {i+1}/{len(missed_files_list)}")
        
        return validated_groups

    def _final_quality_assurance(self, preliminary_groups: List[Dict], all_file_info: Dict) -> List[Dict]:
        """
        Tầng 6: Quality Assurance và Final Validation
        """
        final_groups = []
        total_groups = len(preliminary_groups)
        
        for i, group_data in enumerate(preliminary_groups):
            self._check_pause()
            if not self.is_running: return []
            
            percentage = ((i + 1) / total_groups) * 100
            self.progress_updated.emit(
                i + 1, 
                total_groups, 
                f"Tầng 6/7: QA nhóm {i+1}/{total_groups} ({percentage:.1f}%)"
            )
            
            group_type = group_data.get('type', 'unknown')
            files = group_data.get('files', [])
            
            if len(files) < 2:
                continue
            
            # Enhanced quality checks
            quality_passed = True
            
            # Check 1: Minimum similarity threshold per type
            if group_type == "duplicate":
                # Duplicates should have identical or near-identical hashes
                hashes = set()
                for file_data in files:
                    if 'dhash' in file_data:
                        hashes.add(str(file_data['dhash']))
                
                # Allow for slight hash variations due to compression
                if len(hashes) > min(3, len(files)):  # Too many different hashes
                    quality_passed = False
                    
            elif group_type == "hybrid_subject":
                # Hybrid should have clear composition
                composition = group_data.get('composition', {})
                duplicate_ratio = composition.get('duplicate_ratio', 0)
                
                # Must have some duplicates to be hybrid
                if duplicate_ratio < 0.2:  # Less than 20% duplicates might not be true hybrid
                    # Re-analyze as similar_subject
                    group_data['type'] = 'similar_subject'
                    group_data['analysis_method'] = "V32+: Reclassified from Hybrid to Similar (QA)"
                    
            elif group_type == "similar_subject":
                # Similar subjects should have decent similarity
                score = group_data.get('score', 0)
                if score < 0.65:  # Too low similarity
                    quality_passed = False
            
            # Check 2: File accessibility and validity
            valid_files = []
            for file_data in files:
                file_path = file_data.get('path', '')
                if file_path and file_path in all_file_info:
                    # Verify file still exists and accessible
                    try:
                        import os
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            valid_files.append(file_data)
                    except Exception:
                        continue  # Skip inaccessible files
            
            if len(valid_files) < 2:
                quality_passed = False
            
            # Check 3: Enhanced metadata validation
            if quality_passed and group_type == "hybrid_subject":
                # Ensure action plan is coherent
                action_plan = group_data.get('action_plan', {})
                if action_plan:
                    total_files = len(valid_files)
                    expected_actions = action_plan.get('total_actions', 0)
                    
                    # Action count should make sense
                    if expected_actions >= total_files:  # Can't act on more files than exist
                        # Recalculate action plan
                        composition = group_data.get('composition', {})
                        new_action_plan = self._generate_action_plan(group_type, composition, valid_files)
                        group_data['action_plan'] = new_action_plan
            
            if quality_passed and len(valid_files) >= 2:
                # Update with validated files
                group_data['files'] = valid_files
                group_data['quality_assured'] = True
                group_data['analysis_method'] = group_data.get('analysis_method', '') + " + QA Validated"
                final_groups.append(group_data)
            
        return final_groups

    def _edge_case_detection(self, groups: List[List[str]], all_file_info: Dict) -> List[List[str]]:
        """
        Tầng 7: Edge Case Detection - Phát hiện các trường hợp đặc biệt
        """
        enhanced_groups = []
        
        for group_paths in groups:
            if len(group_paths) < 2:
                continue
                
            self._check_pause()
            if not self.is_running: return []
            
            # Edge Case 1: Very large groups (potential over-clustering)
            if len(group_paths) > 20:  # Groups too large might be incorrect
                # Split large groups using stricter similarity
                sub_groups = self._split_large_group(group_paths, threshold=0.8)
                enhanced_groups.extend(sub_groups)
                continue
            
            # Edge Case 2: Mixed file types in same group
            file_extensions = set()
            for path in group_paths:
                import os
                ext = os.path.splitext(path)[1].lower()
                file_extensions.add(ext)
            
            if len(file_extensions) > 2:  # Too many different file types
                # Group by extension first, then by similarity
                ext_groups = defaultdict(list)
                for path in group_paths:
                    import os
                    ext = os.path.splitext(path)[1].lower()
                    ext_groups[ext].append(path)
                
                for ext_group in ext_groups.values():
                    if len(ext_group) >= 2:
                        enhanced_groups.append(ext_group)
                continue
            
            # Edge Case 3: Files with very different sizes (potential false positives)
            file_sizes = []
            for path in group_paths:
                if path in all_file_info:
                    size = all_file_info[path].get('size', 0)
                    file_sizes.append((path, size))
            
            if file_sizes:
                sizes_only = [size for _, size in file_sizes]
                size_ratio = max(sizes_only) / min(sizes_only) if min(sizes_only) > 0 else 1
                
                if size_ratio > 10:  # Sizes too different (10x difference)
                    # Group by similar sizes
                    size_groups = self._group_by_similar_sizes(file_sizes, tolerance=2.0)
                    enhanced_groups.extend(size_groups)
                    continue
            
            # If passes all edge case checks, keep the group
            enhanced_groups.append(group_paths)
        
        return enhanced_groups

    def _split_large_group(self, large_group: List[str], threshold: float = 0.8) -> List[List[str]]:
        """
        Chia nhóm lớn thành các nhóm nhỏ hơn với similarity cao hơn
        """
        if len(large_group) <= 4:  # Don't split small groups
            return [large_group]
        
        # Use hierarchical clustering for large groups
        sub_groups = []
        remaining_files = large_group.copy()
        
        while len(remaining_files) >= 2:
            # Start new sub-group with first remaining file
            current_subgroup = [remaining_files.pop(0)]
            
            # Add files that are highly similar to current sub-group
            files_to_remove = []
            for file_path in remaining_files:
                # Check similarity with all files in current sub-group
                similarities = []
                for existing_file in current_subgroup:
                    sim = self._calculate_subject_similarity(file_path, existing_file)
                    similarities.append(sim)
                
                avg_sim = sum(similarities) / len(similarities) if similarities else 0
                if avg_sim >= threshold:
                    current_subgroup.append(file_path)
                    files_to_remove.append(file_path)
            
            # Remove added files from remaining
            for file_path in files_to_remove:
                remaining_files.remove(file_path)
            
            # Add sub-group if it has at least 2 files
            if len(current_subgroup) >= 2:
                sub_groups.append(current_subgroup)
            
            # Prevent infinite loop
            if len(files_to_remove) == 0 and len(remaining_files) > 0:
                # Force remove one file to continue
                remaining_files.pop(0)
        
        return sub_groups if sub_groups else [large_group]

    def _group_by_similar_sizes(self, file_size_pairs: List[Tuple[str, int]], tolerance: float = 2.0) -> List[List[str]]:
        """
        Nhóm files theo kích thước tương tự
        """
        size_groups = []
        sorted_pairs = sorted(file_size_pairs, key=lambda x: x[1])  # Sort by size
        
        current_group = []
        current_base_size = 0
        
        for file_path, size in sorted_pairs:
            if not current_group:
                current_group = [file_path]
                current_base_size = size
            else:
                # Check if size is within tolerance of base size
                size_ratio = size / current_base_size if current_base_size > 0 else 1
                if size_ratio <= tolerance:
                    current_group.append(file_path)
                else:
                    # Start new group
                    if len(current_group) >= 2:
                        size_groups.append(current_group)
                    current_group = [file_path]
                    current_base_size = size
        
        # Add last group
        if len(current_group) >= 2:
            size_groups.append(current_group)
        
        return size_groups

    def _comprehensive_duplicate_detection(self, file_list: List[str]) -> Tuple[Dict[str, Dict], Dict[str, List[str]]]:
        """
        Tầng 1: 🏆 COMPREHENSIVE DUPLICATE DETECTION PRIORITY
        Quét trùng lặp toàn diện với multiple algorithms
        """
        all_file_info = {}
        comprehensive_duplicates = defaultdict(list)
        
        total_files = len(file_list)
        processed = 0
        
        # Phase 1: Basic hash detection (nhanh nhất)
        hash_groups = defaultdict(list)
        
        for file_path in file_list:
            self._check_pause()
            if not self.is_running: return {}, {}
            
            processed += 1
            progress_pct = (processed / total_files) * 40  # 40% cho phase 1
            self.progress_updated.emit(
                processed, 
                total_files, 
                f"Tầng 1/7: Hash Detection {processed}/{total_files} ({progress_pct:.1f}%)"
            )
            
            try:
                file_info = self._get_comprehensive_file_info(file_path)
                if file_info:
                    all_file_info[file_path] = file_info
                    
                    # Group by dhash
                    dhash = file_info.get('dhash')
                    if dhash:
                        hash_groups[dhash].append(file_path)
                        
            except Exception as e:
                continue
        
        # Phase 2: Multi-hash verification (chính xác hơn)
        processed = 0
        for dhash, paths in hash_groups.items():
            if len(paths) > 1:  # Potential duplicates
                self._check_pause()
                if not self.is_running: return all_file_info, comprehensive_duplicates
                
                processed += len(paths)
                progress_pct = 40 + (processed / total_files) * 35  # 35% cho phase 2
                self.progress_updated.emit(
                    processed, 
                    len(hash_groups), 
                    f"Tầng 1/7: Multi-Hash Verification ({progress_pct:.1f}%)"
                )
                
                # Verify với multiple hash algorithms
                verified_duplicates = self._verify_duplicates_multi_hash(paths)
                if verified_duplicates and len(verified_duplicates) > 1:
                    comprehensive_duplicates[str(dhash)] = verified_duplicates
        
        # Phase 3: Size-based clustering cho files chưa có trong duplicates
        remaining_files = []
        duplicate_files = set()
        for dup_group in comprehensive_duplicates.values():
            duplicate_files.update(dup_group)
        
        for file_path in all_file_info.keys():
            if file_path not in duplicate_files:
                remaining_files.append(file_path)
        
        if remaining_files:
            size_based_duplicates = self._size_based_duplicate_detection(remaining_files, all_file_info)
            
            # Merge size-based duplicates
            for size_key, size_group in size_based_duplicates.items():
                if len(size_group) > 1:
                    comprehensive_duplicates[f"size_{size_key}"] = size_group
        
        self.progress_updated.emit(total_files, total_files, "Tầng 1/7: ✅ Duplicate Detection Complete!")
        
        return all_file_info, comprehensive_duplicates

    def _get_comprehensive_file_info(self, file_path: str) -> Optional[Dict]:
        """
        Thu thập thông tin file toàn diện cho duplicate detection
        """
        try:
            import os
            from PIL import Image
            
            # Basic file info
            stat = os.stat(file_path)
            file_info = {
                'path': file_path,
                'size': stat.st_size,
                'timestamp': stat.st_mtime,
                'name': os.path.basename(file_path),
                'ext': os.path.splitext(file_path)[1].lower()
            }
            
            # Image-specific info
            with Image.open(file_path) as img:
                file_info['width'] = img.width
                file_info['height'] = img.height
                file_info['mode'] = img.mode
                file_info['format'] = img.format
                
                # Multiple hash algorithms for better accuracy
                rgb_img = img.convert("RGB")
                file_info['dhash'] = imagehash.dhash(rgb_img)
                file_info['phash'] = imagehash.phash(rgb_img)  # Perceptual hash
                file_info['ahash'] = imagehash.average_hash(rgb_img)  # Average hash
                
                # Content signature
                file_info['pixel_signature'] = self._generate_pixel_signature(rgb_img)
                
            return file_info
            
        except Exception as e:
            return None

    def _generate_pixel_signature(self, img: Image.Image) -> str:
        """
        Tạo signature từ pixel data để detect duplicates chính xác hơn
        """
        try:
            # Resize to small size for signature
            small_img = img.resize((8, 8), Image.Resampling.LANCZOS)
            pixels = list(small_img.getdata())
            
            # Create signature from pixel values
            signature = hash(tuple(pixels))
            return str(signature)
        except:
            return ""

    def _verify_duplicates_multi_hash(self, candidate_paths: List[str]) -> List[str]:
        """
        Verify duplicates bằng multiple hash algorithms
        """
        verified_group = []
        
        if len(candidate_paths) < 2:
            return candidate_paths
        
        # Lấy file đầu tiên làm reference
        reference_path = candidate_paths[0]
        verified_group.append(reference_path)
        
        try:
            with Image.open(reference_path) as ref_img:
                ref_rgb = ref_img.convert("RGB")
                ref_dhash = imagehash.dhash(ref_rgb)
                ref_phash = imagehash.phash(ref_rgb)
                ref_ahash = imagehash.average_hash(ref_rgb)
                
                for candidate_path in candidate_paths[1:]:
                    try:
                        with Image.open(candidate_path) as cand_img:
                            cand_rgb = cand_img.convert("RGB")
                            cand_dhash = imagehash.dhash(cand_rgb)
                            cand_phash = imagehash.phash(cand_rgb)
                            cand_ahash = imagehash.average_hash(cand_rgb)
                            
                            # Verify with multiple hashes
                            dhash_diff = ref_dhash - cand_dhash
                            phash_diff = ref_phash - cand_phash
                            ahash_diff = ref_ahash - cand_ahash
                            
                            # Very strict for duplicates
                            if dhash_diff <= 2 and phash_diff <= 3 and ahash_diff <= 3:
                                verified_group.append(candidate_path)
                                
                    except Exception:
                        continue
                        
        except Exception:
            return candidate_paths  # Fallback
        
        return verified_group if len(verified_group) > 1 else []

    def _size_based_duplicate_detection(self, file_paths: List[str], all_file_info: Dict) -> Dict[str, List[str]]:
        """
        Phát hiện duplicates dựa trên kích thước và metadata
        """
        size_groups = defaultdict(list)
        
        for file_path in file_paths:
            file_info = all_file_info.get(file_path, {})
            size = file_info.get('size', 0)
            width = file_info.get('width', 0)
            height = file_info.get('height', 0)
            
            # Create composite key
            size_key = f"{size}_{width}x{height}"
            size_groups[size_key].append(file_path)
        
        # Filter to keep only groups with multiple files
        return {k: v for k, v in size_groups.items() if len(v) > 1}

    def _ultra_strict_quality_control(self, groups: List[List[str]], all_file_info: Dict, similarity_matrix: Optional[Dict]) -> List[List[str]]:
        """
        Tầng 6: ⚡ Ultra-Strict Quality Control với ngưỡng 0.82-0.85
        Đảm bảo chất lượng cao nhất cho similar & hybrid groups
        """
        ultra_strict_groups = []
        recheck_files = set()
        
        total_groups = len(groups)
        for i, group_paths in enumerate(groups):
            self._check_pause()
            if not self.is_running: return []
            
            percentage = ((i + 1) / total_groups) * 100
            self.progress_updated.emit(
                i + 1, 
                total_groups, 
                f"Tầng 6/7: Ultra-Strict QC {i+1}/{total_groups} ({percentage:.1f}%)"
            )
            
            if len(group_paths) < 2:
                continue
            
            # Phân loại group type trước
            group_type = self._determine_group_type_preliminary(group_paths, all_file_info)
            
            if group_type == "duplicate":
                # Duplicates luôn pass (đã được verify ở tầng 1)
                ultra_strict_groups.append(group_paths)
                continue
            
            # Áp dụng ultra-strict threshold cho similar & hybrid
            ultra_strict_scores = []
            for j in range(len(group_paths)):
                for k in range(j + 1, len(group_paths)):
                    score = self._calculate_subject_similarity(group_paths[j], group_paths[k])
                    ultra_strict_scores.append(score)
            
            avg_score = sum(ultra_strict_scores) / len(ultra_strict_scores) if ultra_strict_scores else 0.0
            
            # 🎯 ANTI-GROUPING MISTAKE THRESHOLDS (Cao hơn để tránh gộp nhầm)
            if group_type == "hybrid_subject":
                ultra_threshold = 0.88  # Rất cao cho hybrid (tránh gộp nhầm)
            else:  # similar_subject
                ultra_threshold = 0.92  # Cực kỳ cao cho similar (chỉ gộp khi thực sự tương tự)
            
            if avg_score >= ultra_threshold:
                # ✅ PASS - Ultra high quality
                ultra_strict_groups.append(group_paths)
            else:
                # ❌ FAIL - Cần phân tách hoặc loại bỏ
                # Thử tách thành sub-groups với ultra-strict criteria
                ultra_sub_groups = self._create_ultra_strict_subgroups(group_paths, ultra_threshold)
                
                if ultra_sub_groups:
                    ultra_strict_groups.extend(ultra_sub_groups)
                else:
                    # Không tách được - đưa vào recheck
                    recheck_files.update(group_paths)
        
        # Re-process các files bị recheck với tiêu chí nghiêm ngặt
        if len(recheck_files) >= 2:
            recheck_list = list(recheck_files)
            self.progress_updated.emit(
                total_groups, 
                total_groups, 
                f"Tầng 6/7: Re-processing {len(recheck_list)} files..."
            )
            
            recovery_groups = self._ultra_strict_recovery(recheck_list)
            ultra_strict_groups.extend(recovery_groups)
        
        return ultra_strict_groups

    def _determine_group_type_preliminary(self, group_paths: List[str], all_file_info: Dict) -> str:
        """
        Xác định loại group CHÍNH XÁC:
        - Trùng lặp: Files giống hệt nhau (hash giống)
        - Hỗn hợp: Có cả trùng lặp VÀ tương tự (chủ thể + màu sắc)
        - Tương tự: Chỉ tương tự về chủ thể và màu sắc
        """
        # Analyze hash distribution cho duplicate detection
        hash_groups = defaultdict(list)
        for path in group_paths:
            file_info = all_file_info.get(path, {})
            dhash = file_info.get('dhash')
            if dhash:
                hash_groups[dhash].append(path)
        
        # Đếm số files trùng lặp thực sự (cùng hash)
        duplicate_files = []
        for dhash, paths in hash_groups.items():
            if len(paths) > 1:
                duplicate_files.extend(paths)
        
        total_files = len(group_paths)
        duplicate_count = len(duplicate_files)
        similar_only_count = total_files - duplicate_count
        
        # Logic phân loại CHÍNH XÁC
        if duplicate_count == total_files:
            # TẤT CẢ files đều trùng lặp
            return "duplicate"
        elif duplicate_count > 0 and similar_only_count > 0:
            # CÓ CẢ trùng lặp VÀ tương tự → HYBRID
            return "hybrid_subject"
        else:
            # CHỈ có files tương tự (không có trùng lặp)
            return "similar_subject"

    def _create_ultra_strict_subgroups(self, group_paths: List[str], threshold: float) -> List[List[str]]:
        """
        Tạo sub-groups với tiêu chí ultra-strict
        """
        subgroups = []
        remaining_files = group_paths.copy()
        
        while len(remaining_files) >= 2:
            # Bắt đầu với file có nhiều connections tốt nhất
            best_starter = self._find_best_starter_file(remaining_files, threshold)
            current_subgroup = [best_starter]
            remaining_files.remove(best_starter)
            
            # Thêm files tương thích với ultra-strict criteria
            files_to_remove = []
            for file_path in remaining_files:
                # Kiểm tra compatibility với TOÀN BỘ subgroup
                compatible = True
                compatibility_scores = []
                
                for existing_file in current_subgroup:
                    score = self._calculate_subject_similarity(file_path, existing_file)
                    compatibility_scores.append(score)
                    if score < threshold:
                        compatible = False
                        break
                
                if compatible and len(compatibility_scores) > 0:
                    # Đảm bảo avg score với subgroup cũng đạt threshold
                    avg_compatibility = sum(compatibility_scores) / len(compatibility_scores)
                    if avg_compatibility >= threshold:
                        current_subgroup.append(file_path)
                        files_to_remove.append(file_path)
            
            # Remove files đã thêm
            for file_path in files_to_remove:
                remaining_files.remove(file_path)
            
            # Thêm subgroup nếu đủ lớn và chất lượng cao
            if len(current_subgroup) >= 2:
                # Double-check quality của subgroup
                subgroup_quality = self._calculate_group_internal_similarity(current_subgroup)
                if subgroup_quality >= threshold:
                    subgroups.append(current_subgroup)
            
            # Tránh vòng lặp vô tận
            if len(files_to_remove) == 0 and len(remaining_files) > 0:
                remaining_files.pop(0)  # Remove file không tương thích
        
        return subgroups

    def _find_best_starter_file(self, file_paths: List[str], threshold: float) -> str:
        """
        Tìm file tốt nhất để bắt đầu subgroup (có nhiều connections chất lượng cao)
        """
        best_file = file_paths[0]
        best_score = 0
        
        for candidate in file_paths:
            high_quality_connections = 0
            total_score = 0
            
            for other_file in file_paths:
                if candidate != other_file:
                    score = self._calculate_subject_similarity(candidate, other_file)
                    if score >= threshold:
                        high_quality_connections += 1
                        total_score += score
            
            # Ưu tiên file có nhiều connections chất lượng cao
            final_score = high_quality_connections * 1000 + total_score
            if final_score > best_score:
                best_score = final_score
                best_file = candidate
        
        return best_file

    def _ultra_strict_recovery(self, recheck_files: List[str]) -> List[List[str]]:
        """
        Recovery với tiêu chí ultra-strict cho files bị recheck
        """
        recovery_groups = []
        remaining_files = recheck_files.copy()
        
        # Sử dụng threshold thấp hơn một chút cho recovery (0.80)
        recovery_threshold = 0.80
        
        while len(remaining_files) >= 2:
            # Tìm cặp tốt nhất
            best_pair = None
            best_score = 0
            
            for i in range(len(remaining_files)):
                for j in range(i + 1, len(remaining_files)):
                    score = self._calculate_subject_similarity(remaining_files[i], remaining_files[j])
                    if score >= recovery_threshold and score > best_score:
                        best_score = score
                        best_pair = (remaining_files[i], remaining_files[j])
            
            if best_pair:
                # Tạo recovery group từ best pair
                recovery_group = list(best_pair)
                remaining_files.remove(best_pair[0])
                remaining_files.remove(best_pair[1])
                
                # Thử mở rộng recovery group
                files_to_remove = []
                for file_path in remaining_files:
                    compatible = True
                    for existing_file in recovery_group:
                        score = self._calculate_subject_similarity(file_path, existing_file)
                        if score < recovery_threshold:
                            compatible = False
                            break
                    
                    if compatible:
                        recovery_group.append(file_path)
                        files_to_remove.append(file_path)
                
                for file_path in files_to_remove:
                    remaining_files.remove(file_path)
                
                if len(recovery_group) >= 2:
                    recovery_groups.append(recovery_group)
            else:
                break  # Không tìm được cặp nào tốt
        
        return recovery_groups

    def _find_new_groups_from_recheck(self, recheck_files: List[str], strict_threshold: float = 0.75) -> List[List[str]]:
        """
        Tìm các nhóm mới từ files bị recheck
        """
        new_groups = []
        remaining_files = recheck_files.copy()
        
        while len(remaining_files) >= 2:
            # Bắt đầu với file đầu tiên
            current_group = [remaining_files.pop(0)]
            
            # Tìm files tương thích
            files_to_remove = []
            for file_path in remaining_files:
                # Kiểm tra với tất cả files trong current_group
                compatible = True
                for existing_file in current_group:
                    score = self._calculate_subject_similarity(file_path, existing_file)
                    if score < strict_threshold:
                        compatible = False
                        break
                
                if compatible:
                    current_group.append(file_path)
                    files_to_remove.append(file_path)
            
            # Remove files đã thêm vào group
            for file_path in files_to_remove:
                remaining_files.remove(file_path)
            
            # Thêm group nếu có ít nhất 2 files
            if len(current_group) >= 2:
                new_groups.append(current_group)
            
            # Tránh vòng lặp vô tận
            if len(files_to_remove) == 0 and len(remaining_files) > 0:
                remaining_files.pop(0)  # Loại bỏ file không tương thích
        
        return new_groups

    def _final_validation_and_cross_optimize(self, groups: List[List[str]], all_file_info: Dict) -> List[Dict]:
        """
        Tầng 7: Final Validation & Cross-Optimization
        Tối ưu chéo cuối cùng: quét lại các nhóm để phân tách nếu cần
        """
        optimized_groups = []
        total_groups = len(groups)
        
        for i, group_paths in enumerate(groups):
            self._check_pause()
            if not self.is_running: return []
            
            percentage = ((i + 1) / total_groups) * 100
            self.progress_updated.emit(
                i + 1, 
                total_groups, 
                f"Tầng 7/7: Tối ưu chéo {i+1}/{total_groups} ({percentage:.1f}%)"
            )
            
            if len(group_paths) < 2:
                continue
            
            # Cross-optimization: Kiểm tra xem nhóm có thể tách thành 2 nhóm tốt hơn không
            if len(group_paths) >= 4:  # Chỉ tách nhóm đủ lớn
                potential_split = self._attempt_group_split_optimization(group_paths)
                if potential_split:  # Nếu tách được thành nhóm tốt hơn
                    for split_group in potential_split:
                        if len(split_group) >= 2:
                            group_data = self._create_final_group_data(split_group, all_file_info, "Cross-Optimized Split")
                            if group_data:
                                optimized_groups.append(group_data)
                    continue
            
            # Nếu không tách được hoặc không cần tách, tạo group data thông thường
            group_data = self._create_final_group_data(group_paths, all_file_info, "Cross-Validated")
            if group_data:
                optimized_groups.append(group_data)
        
        return optimized_groups

    def _attempt_group_split_optimization(self, group_paths: List[str]) -> Optional[List[List[str]]]:
        """
        Thử tách nhóm thành 2 nhóm con tối ưu hơn
        """
        if len(group_paths) < 4:
            return None
        
        best_split = None
        best_split_quality = 0.0
        
        # Thử các cách tách khác nhau
        for split_point in range(2, len(group_paths) - 1):
            # Tách thành 2 nhóm theo similarity matrix
            similarity_scores = []
            for i in range(len(group_paths)):
                for j in range(i + 1, len(group_paths)):
                    score = self._calculate_subject_similarity(group_paths[i], group_paths[j])
                    similarity_scores.append((i, j, score))
            
            # Sắp xếp theo similarity giảm dần
            similarity_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Thử clustering thành 2 groups
            group1 = [group_paths[0]]
            group2 = []
            assigned = {0}
            
            for i, j, score in similarity_scores:
                if len(group1) >= split_point:
                    if i not in assigned and j not in assigned:
                        group2.append(group_paths[i])
                        group2.append(group_paths[j])
                        assigned.update([i, j])
                    elif i not in assigned:
                        group2.append(group_paths[i])
                        assigned.add(i)
                    elif j not in assigned:
                        group2.append(group_paths[j])
                        assigned.add(j)
                else:
                    if i not in assigned and j not in assigned:
                        group1.append(group_paths[i])
                        group1.append(group_paths[j])
                        assigned.update([i, j])
                    elif i not in assigned:
                        group1.append(group_paths[i])
                        assigned.add(i)
                    elif j not in assigned:
                        group1.append(group_paths[j])
                        assigned.add(j)
            
            # Assign remaining files
            for idx, file_path in enumerate(group_paths):
                if idx not in assigned:
                    if len(group1) <= len(group2):
                        group1.append(file_path)
                    else:
                        group2.append(file_path)
            
            # Đánh giá chất lượng split
            if len(group1) >= 2 and len(group2) >= 2:
                quality1 = self._calculate_group_internal_similarity(group1)
                quality2 = self._calculate_group_internal_similarity(group2)
                split_quality = (quality1 + quality2) / 2
                
                # So sánh với chất lượng nhóm gốc
                original_quality = self._calculate_group_internal_similarity(group_paths)
                
                if split_quality > original_quality + 0.05:  # Phải tốt hơn ít nhất 5%
                    if split_quality > best_split_quality:
                        best_split_quality = split_quality
                        best_split = [group1, group2]
        
        return best_split

    def _calculate_group_internal_similarity(self, group_paths: List[str]) -> float:
        """
        Tính similarity trung bình trong nhóm
        """
        if len(group_paths) < 2:
            return 0.0
        
        scores = []
        for i in range(len(group_paths)):
            for j in range(i + 1, len(group_paths)):
                score = self._calculate_subject_similarity(group_paths[i], group_paths[j])
                scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0

    def _create_final_group_data(self, group_paths: List[str], all_file_info: Dict, optimization_method: str) -> Optional[Dict]:
        """
        Tạo group data cuối cùng với thông tin đầy đủ
        """
        if len(group_paths) < 2:
            return None
        
        # Analyze hash distribution for classification
        hashes = defaultdict(list)
        for path in group_paths:
            file_info = all_file_info.get(path)
            if file_info and 'dhash' in file_info:
                hashes[file_info['dhash']].append(path)
        
        num_duplicates = sum(1 for paths in hashes.values() if len(paths) > 1)
        num_unique_in_group = len(hashes)
        
        # Enhanced group classification
        if num_unique_in_group == 1:
            group_type = "duplicate"
        elif num_duplicates > 0:
            group_type = "hybrid_subject"
        else:
            group_type = "similar_subject"
        
        # Calculate average similarity
        scores = []
        for i in range(len(group_paths)):
            for j in range(i + 1, len(group_paths)):
                score = self._calculate_subject_similarity(group_paths[i], group_paths[j])
                scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Enhanced quality thresholds
        strict_thresholds = {
            "duplicate": 0.0,       # Duplicates always pass
            "hybrid_subject": 0.72, # Stricter for hybrid
            "similar_subject": 0.76  # Very strict for similar
        }
        
        min_score = strict_thresholds.get(group_type, 0.76)
        if group_type != "duplicate" and avg_score < min_score:
            return None  # Skip low-quality groups
        
        # Enhanced file info
        enhanced_files = []
        duplicate_paths = []
        similar_paths = []
        
        # Categorize files
        for hash_val, hash_group in hashes.items():
            if len(hash_group) > 1:
                duplicate_paths.extend(hash_group)
            else:
                similar_paths.extend(hash_group)
        
        for path in group_paths:
            if path in all_file_info:
                file_data = all_file_info[path].copy()
                # Ensure path is available and normalized
                file_data['path'] = os.path.normpath(path)
                if 'dhash' in file_data:
                    file_data['hash_str'] = str(file_data['dhash'])
                
                # Enhanced role assignment
                if path in duplicate_paths:
                    file_data['role_in_group'] = 'duplicate'
                    file_data['action_suggestion'] = 'delete_candidate'
                elif path in similar_paths:
                    file_data['role_in_group'] = 'similar'
                    file_data['action_suggestion'] = 'rename_candidate'
                else:
                    file_data['role_in_group'] = 'primary'
                    file_data['action_suggestion'] = 'keep'
                
                enhanced_files.append(file_data)
        
        # Generate metadata
        metadata = {
            'duplicate_files': len(duplicate_paths),
            'similar_only_files': len(similar_paths),
            'duplicate_paths': duplicate_paths,
            'similar_only_paths': similar_paths,
            'composition_type': 'pure_duplicate' if group_type == 'duplicate' else 
                               ('hybrid' if group_type == 'hybrid_subject' else 'pure_similar')
        }
        
        # Enhanced group info
        group_info = {
            "type": group_type,
            "files": enhanced_files,
            "score": 1.0 if group_type == "duplicate" else avg_score,
            "analysis_method": f"V33+: {optimization_method} - Ultra Strict ({group_type.replace('_', ' ').title()})",
            "composition": metadata,
            "action_plan": self._generate_action_plan(group_type, metadata, enhanced_files),
            "quality_level": "ultra_strict",
            "optimization_method": optimization_method
        }
        
        return group_info

    def _optimize_for_large_dataset(self, file_count: int):
        """
        Tối ưu với ANTI-GROUPING MISTAKE THRESHOLDS (0.88-0.92)
        Tránh gộp nhầm là ưu tiên hàng đầu
        """
        if file_count > 10000:
            # Very large dataset - ANTI-MISTAKE MODE
            self.batch_size = min(6, self.batch_size)   # Rất nhỏ cho precision tối đa
            self.similarity_threshold = 0.92  # 🎯 ANTI-MISTAKE threshold  
            self.duplicate_threshold = 0.99   # Gần như perfect cho duplicates
            self.strict_mode = True
            self.anti_mistake_mode = True
        elif file_count > 5000:
            # Large dataset - HIGH PRECISION
            self.batch_size = min(12, self.batch_size)
            self.similarity_threshold = 0.90  # 🎯 HIGH PRECISION threshold
            self.duplicate_threshold = 0.98
            self.strict_mode = True
            self.anti_mistake_mode = True
        elif file_count > 1000:
            # Medium dataset - CAREFUL MODE
            self.batch_size = min(20, self.batch_size)
            self.similarity_threshold = 0.88  # 🎯 CAREFUL threshold  
            self.duplicate_threshold = 0.96
            self.strict_mode = True
            self.anti_mistake_mode = True
        else:
            # Small dataset - STILL CAREFUL
            self.similarity_threshold = 0.85  # Vẫn cao để tránh gộp nhầm
            self.duplicate_threshold = 0.94
            self.strict_mode = True
            self.anti_mistake_mode = False

        # Enhanced ANTI-MISTAKE controls
        self.enable_cross_optimization = True
        self.enable_split_detection = file_count > 100  # Enable sớm hơn
        self.max_group_size = 15 if file_count > 1000 else 25  # Nhóm nhỏ hơn
        self.enable_multi_hash_verification = True
        
        # Anti-mistake specific settings
        if hasattr(self, 'anti_mistake_mode') and self.anti_mistake_mode:
            self.enable_color_detail_focus = True  # Tập trung màu sắc + chi tiết
            self.enable_strong_penalties = True    # Penalty mạnh
            self.min_color_similarity = 0.75       # Màu sắc tối thiểu
            self.min_detail_similarity = 0.70      # Chi tiết tối thiểu

    def stop(self): self.is_running = False; self.resume()
    def pause(self): self.mutex.lock(); self._is_paused = True; self.mutex.unlock()
    def resume(self): self.mutex.lock(); self._is_paused = False; self.mutex.unlock(); self.pause_condition.wakeAll()

    def _enhanced_rescan_validation(self, final_groups: List[Dict], all_file_info: Dict) -> List[Dict]:
        """
        Tầng 8: Enhanced Re-scan Validation để phát hiện các file tương tự bị bỏ sót
        """
        print("DEBUG: Starting enhanced rescan validation...")
        
        # Lấy tất cả files đã được grouped
        grouped_files = set()
        for group_dict in final_groups:
            files = group_dict.get('files', [])
            for file_info in files:
                if isinstance(file_info, dict) and 'path' in file_info:
                    grouped_files.add(file_info['path'])
        
        # Lấy files chưa được grouped
        all_files = set(all_file_info.keys())
        ungrouped_files = all_files - grouped_files
        
        print(f"DEBUG: Found {len(ungrouped_files)} ungrouped files to re-check")
        
        if len(ungrouped_files) < 2:
            return final_groups
        
        # Thực hiện quét lại với threshold thấp hơn để phát hiện các file bị bỏ sót
        enhanced_groups = []
        ungrouped_list = list(ungrouped_files)
        
        for i in range(len(ungrouped_list)):
            for j in range(i + 1, len(ungrouped_list)):
                file1, file2 = ungrouped_list[i], ungrouped_list[j]
                
                # Tính similarity với threshold thấp hơn để bắt được những case bị bỏ sót
                similarity_score = self._calculate_subject_similarity(file1, file2)
                
                # Threshold thấp hơn để phát hiện similar bị bỏ sót (0.80 thay vì 0.92)
                if similarity_score >= 0.80:
                    print(f"DEBUG: Found missed similarity: {os.path.basename(file1)} <-> {os.path.basename(file2)} (score: {similarity_score:.3f})")
                    
                    # Tìm hoặc tạo group cho cặp file này
                    group_found = False
                    for group in enhanced_groups:
                        if file1 in group or file2 in group:
                            if file1 not in group:
                                group.append(file1)
                            if file2 not in group:
                                group.append(file2)
                            group_found = True
                            break
                    
                    if not group_found:
                        enhanced_groups.append([file1, file2])
        
        # Merge similar groups (nếu có files chung)
        merged_enhanced_groups = []
        for new_group in enhanced_groups:
            merged = False
            for existing_group in merged_enhanced_groups:
                if any(f in existing_group for f in new_group):
                    # Merge groups
                    for f in new_group:
                        if f not in existing_group:
                            existing_group.append(f)
                    merged = True
                    break
            
            if not merged:
                merged_enhanced_groups.append(new_group)
        
        print(f"DEBUG: Enhanced rescan found {len(merged_enhanced_groups)} additional groups")
        
        # Convert enhanced groups to proper format and add to final results
        additional_groups = []
        for group_paths in merged_enhanced_groups:
            if len(group_paths) >= 2:
                group_data = self._create_final_group_data(group_paths, all_file_info, "enhanced_rescan")
                if group_data:
                    additional_groups.append(group_data)
        
        return final_groups + additional_groups
