# auto_processor.py
# Logic tự động xử lý file trùng lặp và tương tự

import os
import shutil
from typing import List, Dict, Any, Tuple, Optional
from send2trash import send2trash

class AutoProcessor:
    """Xử lý tự động file trùng lặp và tương tự"""
    
    def __init__(self):
        self.processed_count = 0
        self.deleted_count = 0
        self.renamed_count = 0
        self.errors = []
    
    def process_all_groups(self, groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Xử lý tất cả các nhóm"""
        self.processed_count = 0
        self.deleted_count = 0
        self.renamed_count = 0
        self.errors = []
        
        duplicate_groups = []
        similar_groups = []
        
        # Phân loại nhóm
        for group in groups:
            if group.get('type') == 'duplicate':
                duplicate_groups.append(group)
            else:
                similar_groups.append(group)
        
        # Xử lý file trùng lặp
        for group in duplicate_groups:
            try:
                self._process_duplicate_group(group)
            except Exception as e:
                self.errors.append(f"Lỗi xử lý nhóm trùng lặp: {e}")
        
        # Xử lý file tương tự
        for i, group in enumerate(similar_groups, 1):
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
        """Xử lý nhóm file trùng lặp - giữ lại file tốt nhất"""
        files = group.get('files', [])
        if len(files) < 2:
            return
        
        # Chuyển đổi files thành danh sách đường dẫn với validation tốt hơn
        file_paths = []
        for file_info in files:
            path = None
            
            if isinstance(file_info, dict):
                path = file_info.get('path', '')
            elif isinstance(file_info, str):
                path = file_info
            else:
                continue
            
            # Normalize path và kiểm tra tồn tại
            if path:
                try:
                    normalized_path = os.path.normpath(os.path.abspath(path))
                    if os.path.exists(normalized_path) and os.path.isfile(normalized_path):
                        file_paths.append(normalized_path)
                        print(f"📁 Tìm thấy file: {os.path.basename(normalized_path)}")
                    else:
                        print(f"⚠️ File không tồn tại: {path}")
                except Exception as e:
                    print(f"❌ Lỗi xử lý đường dẫn {path}: {e}")
        
        if len(file_paths) < 2:
            print(f"⚠️ Nhóm trùng lặp chỉ có {len(file_paths)} file hợp lệ, bỏ qua")
            return
        
        # Tìm file tốt nhất để giữ lại
        best_file = self._find_best_file_from_paths(file_paths)
        if not best_file:
            print("❌ Không thể xác định file tốt nhất")
            return
        
        print(f"🏆 File tốt nhất: {os.path.basename(best_file)}")
        
        # Xóa các file còn lại
        deleted_files = []
        for file_path in file_paths:
            if file_path != best_file:
                try:
                    # Kiểm tra lần cuối trước khi xóa
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        send2trash(file_path)
                        self.deleted_count += 1
                        deleted_files.append(os.path.basename(file_path))
                        print(f"✅ Đã xóa: {os.path.basename(file_path)}")
                    else:
                        print(f"⚠️ File đã không tồn tại: {os.path.basename(file_path)}")
                except Exception as e:
                    error_msg = f"Không thể xóa {os.path.basename(file_path)}: {str(e)}"
                    self.errors.append(error_msg)
                    print(f"❌ {error_msg}")
        
        if deleted_files:
            print(f"🔄 Nhóm trùng lặp: Giữ lại {os.path.basename(best_file)}, đã xóa {len(deleted_files)} file")
        else:
            print(f"⚠️ Không xóa được file nào trong nhóm trùng lặp")
    
    def _process_similar_group(self, group: Dict[str, Any], group_number: int):
        """Xử lý nhóm file tương tự - đổi tên theo pattern"""
        files = group.get('files', [])
        if len(files) < 2:
            return
        
        # Chuyển đổi files thành danh sách đường dẫn
        file_paths = []
        for file_info in files:
            if isinstance(file_info, dict):
                path = file_info.get('path', '')
            else:
                path = str(file_info)
            
            if path and os.path.exists(path):
                file_paths.append(path)
        
        # Đổi tên các file trong nhóm
        for i, file_path in enumerate(file_paths, 1):
            try:
                new_path = self._generate_new_name(file_path, group_number, i)
                if new_path != file_path:
                    os.rename(file_path, new_path)
                    self.renamed_count += 1
                    print(f"📝 Đổi tên: {os.path.basename(file_path)} → {os.path.basename(new_path)}")
            except Exception as e:
                self.errors.append(f"Không thể đổi tên {os.path.basename(file_path)}: {e}")
        
        print(f"🎯 Nhóm tương tự {group_number}: Đã đổi tên {len(file_paths)} files")
    
    def _find_best_file_from_paths(self, file_paths: List[str]) -> 'Optional[str]':
        """Tìm file tốt nhất để giữ lại dựa trên tiêu chí"""
        best_file = None
        best_score = -1
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            
            score = self._calculate_file_score(file_path)
            if score > best_score:
                best_score = score
                best_file = file_path
        
        if best_file:
            return best_file
        elif file_paths:
            return file_paths[0]
        else:
            return None
    
    def _calculate_file_score(self, file_path: str) -> float:
        """Tính điểm cho file dựa trên các tiêu chí"""
        score = 0.0
        
        try:
            # Tiêu chí 1: Kích thước file (lớn hơn = tốt hơn)
            file_size = os.path.getsize(file_path)
            score += file_size / (1024 * 1024)  # MB
            
            # Tiêu chí 2: Độ dài tên file (dài hơn = chi tiết hơn)
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            score += len(name_without_ext) * 0.1
            
            # Tiêu chí 3: Không có số trong tên (ưu tiên file gốc)
            if not any(char.isdigit() for char in name_without_ext):
                score += 5.0
            
            # Tiêu chí 4: Không có từ khóa copy, duplicate
            lower_name = name_without_ext.lower()
            if 'copy' not in lower_name and 'duplicate' not in lower_name and 'dup' not in lower_name:
                score += 3.0
            
            # Tiêu chí 5: Định dạng file (ưu tiên JPG > PNG > khác)
            ext = os.path.splitext(file_path)[1].lower()
            try:
                from speed_config import SpeedConfig
                jpg_exts = {'.jpg', '.jpeg'}
                if ext in jpg_exts:
                    score += 2.0
                elif ext in {'.png'}:
                    score += 1.0
                # Give slight bonus for supported image formats
                if ext in set(SpeedConfig.SUPPORTED_IMAGE_EXTENSIONS):
                    score += 0.5
            except Exception:
                if ext in ['.jpg', '.jpeg']:
                    score += 2.0
                elif ext == '.png':
                    score += 1.0
            
        except Exception:
            score = 0.0
        
        return score
    
    def _generate_new_name(self, file_path: str, group_number: int, file_index: int) -> str:
        """Tạo tên mới cho file theo pattern group(index)"""
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        
        # Tạo tên mới theo pattern: group_number(file_index)
        new_name = f"{group_number}({file_index}){ext}"
        new_path = os.path.join(directory, new_name)
        
        # Kiểm tra xem tên mới đã tồn tại chưa
        counter = 1
        while os.path.exists(new_path):
            new_name = f"{group_number}({file_index})_{counter}{ext}"
            new_path = os.path.join(directory, new_name)
            counter += 1
        
        return new_path
    
    def get_summary(self) -> str:
        """Lấy tóm tắt kết quả xử lý"""
        summary = f"""
🎉 TỰ ĐỘNG XỬ LÝ HOÀN TẤT

✅ Kết quả:
   • Đã xóa: {self.deleted_count} file trùng lặp
   • Đã đổi tên: {self.renamed_count} file tương tự
   • Tổng xử lý: {self.deleted_count + self.renamed_count} file

"""
        
        if self.errors:
            summary += f"⚠️ Lỗi ({len(self.errors)}):\n"
            for error in self.errors[:5]:  # Hiển thị tối đa 5 lỗi
                summary += f"   • {error}\n"
            if len(self.errors) > 5:
                summary += f"   • ... và {len(self.errors) - 5} lỗi khác\n"
        
        return summary.strip()

# Global processor instance
auto_processor = AutoProcessor()
