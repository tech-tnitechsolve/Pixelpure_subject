# auto_processor.py
#
# Phiên bản V27: Sửa lỗi Type Hint & Tăng tính ổn định.
# - [FIX] Sửa lỗi Pylance `reportReturnType` bằng cách thay đổi kiểu trả về của
#   `_find_best_file_from_paths` thành `Optional[str]`.
# - [IMPROVED] Thêm các bước kiểm tra `None` sau khi gọi hàm `_find_best_file_from_paths`
#   để đảm bảo chương trình không gặp lỗi khi không tìm thấy file tốt nhất.

import os
import shutil
from typing import List, Dict, Any, Tuple, Optional
from send2trash import send2trash
import imagehash
from PIL import Image
from collections import defaultdict

class AutoProcessor:
    def __init__(self):
        self.processed_count = 0
        self.deleted_count = 0
        self.renamed_count = 0
        self.errors = []
    
    def process_all_groups(self, groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.processed_count = 0
        self.deleted_count = 0
        self.renamed_count = 0
        self.errors = []
        
        duplicate_groups = []
        similar_groups = []
        hybrid_groups = []
        
        for group in groups:
            group_type = group.get('type')
            if group_type == 'duplicate':
                duplicate_groups.append(group)
            elif group_type == 'similar_subject':
                similar_groups.append(group)
            elif group_type == 'hybrid_subject':
                hybrid_groups.append(group)
        
        for group in duplicate_groups:
            try:
                self._process_duplicate_group(group)
            except Exception as e:
                self.errors.append(f"Lỗi xử lý nhóm trùng lặp: {e}")
        
        for i, group in enumerate(hybrid_groups, 1):
            try:
                self._process_hybrid_group(group, i)
            except Exception as e:
                self.errors.append(f"Lỗi xử lý nhóm hỗn hợp {i}: {e}")

        for i, group in enumerate(similar_groups, len(hybrid_groups) + 1):
            try:
                self._process_similar_group(group, i)
            except Exception as e:
                self.errors.append(f"Lỗi xử lý nhóm tương tự {i}: {e}")
        
        return {
            'processed_groups': len(groups),
            'deleted_files': self.deleted_count,
            'renamed_files': self.renamed_count,
            'errors': self.errors
        }
    
    def _process_duplicate_group(self, group: Dict[str, Any]):
        files = group.get('files', [])
        if len(files) < 2: return
        
        file_paths = [f['path'] for f in files if isinstance(f, dict) and 'path' in f and os.path.exists(f['path'])]
        if len(file_paths) < 2: return
        
        best_file = self._find_best_file_from_paths(file_paths)
        if not best_file: return # Kiểm tra nếu không tìm thấy file
        
        for file_path in file_paths:
            if file_path != best_file:
                try:
                    send2trash(file_path)
                    self.deleted_count += 1
                except Exception as e:
                    self.errors.append(f"Không thể xóa {os.path.basename(file_path)}: {e}")

    def _process_similar_group(self, group: Dict[str, Any], group_number: int):
        files = group.get('files', [])
        if len(files) < 2: return

        file_paths = [f['path'] for f in files if isinstance(f, dict) and 'path' in f and os.path.exists(f['path'])]
        file_paths.sort()

        for i, file_path in enumerate(file_paths, 1):
            try:
                new_path = self._generate_new_name(file_path, group_number, i)
                if new_path != file_path:
                    os.rename(file_path, new_path)
                    self.renamed_count += 1
            except Exception as e:
                self.errors.append(f"Không thể đổi tên {os.path.basename(file_path)}: {e}")

    def _process_hybrid_group(self, group: Dict[str, Any], group_number: int):
        files = group.get('files', [])
        if len(files) < 2: return

        file_paths = [f['path'] for f in files if isinstance(f, dict) and 'path' in f and os.path.exists(f['path'])]
        
        hash_groups = defaultdict(list)
        for path in file_paths:
            try:
                with Image.open(path) as img:
                    h = imagehash.dhash(img)
                    hash_groups[h].append(path)
            except Exception as e:
                self.errors.append(f"Không thể hash file {os.path.basename(path)}: {e}")
        
        files_to_keep = []
        for h, paths_in_hash_group in hash_groups.items():
            if len(paths_in_hash_group) > 1:
                best_file = self._find_best_file_from_paths(paths_in_hash_group)
                if best_file: # Kiểm tra nếu tìm thấy file
                    files_to_keep.append(best_file)
                    for path_to_delete in paths_in_hash_group:
                        if path_to_delete != best_file:
                            try:
                                send2trash(path_to_delete)
                                self.deleted_count += 1
                            except Exception as e:
                                self.errors.append(f"Không thể xóa {os.path.basename(path_to_delete)}: {e}")
            else:
                files_to_keep.append(paths_in_hash_group[0])

        files_to_keep.sort()
        for i, file_path in enumerate(files_to_keep, 1):
            try:
                new_path = self._generate_new_name(file_path, group_number, i)
                if new_path != file_path:
                    os.rename(file_path, new_path)
                    self.renamed_count += 1
            except Exception as e:
                self.errors.append(f"Không thể đổi tên {os.path.basename(file_path)}: {e}")

    def _find_best_file_from_paths(self, file_paths: List[str]) -> Optional[str]:
        best_file: Optional[str] = None
        best_score = -1
        
        for file_path in file_paths:
            if not os.path.exists(file_path): continue
            
            score = 0.0
            try:
                score += os.path.getsize(file_path)
                with Image.open(file_path) as img:
                    score += img.width * img.height
                
                filename = os.path.splitext(os.path.basename(file_path))[0]
                if not any(char.isdigit() for char in filename) and 'copy' not in filename.lower():
                    score *= 1.5
                score += len(filename) * 1000

            except Exception:
                score = 0.0

            if score > best_score:
                best_score = score
                best_file = file_path
        
        return best_file
    
    def _generate_new_name(self, file_path: str, group_number: int, file_index: int) -> str:
        directory = os.path.dirname(file_path)
        _, ext = os.path.splitext(file_path)
        
        new_name = f"{group_number}({file_index}){ext}"
        new_path = os.path.join(directory, new_name)
        
        counter = 1
        while os.path.exists(new_path):
            new_name = f"{group_number}({file_index})_{counter}{ext}"
            new_path = os.path.join(directory, new_name)
            counter += 1
        
        return new_path
    
    def get_summary(self) -> str:
        summary = f"🎉 TỰ ĐỘNG XỬ LÝ HOÀN TẤT\n\n✅ Kết quả:\n   • Đã xóa: {self.deleted_count} file\n   • Đã đổi tên: {self.renamed_count} file\n   • Tổng xử lý: {self.deleted_count + self.renamed_count} file"
        if self.errors:
            summary += f"\n\n⚠️ Lỗi ({len(self.errors)}):\n"
            for error in self.errors[:5]:
                summary += f"   • {error}\n"
            if len(self.errors) > 5:
                summary += f"   • ... và {len(self.errors) - 5} lỗi khác"
        return summary.strip()

auto_processor = AutoProcessor()
