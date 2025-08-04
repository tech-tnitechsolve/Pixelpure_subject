# cache_manager.py
# Quản lý cache và cleanup khi tắt ứng dụng

import os
import shutil
import tempfile
import gc
from typing import List
import torch

class CacheManager:
    """Quản lý cache và cleanup tài nguyên"""
    
    def __init__(self):
        self.temp_dirs = []
        self.cache_dirs = []
        
    def add_temp_dir(self, path: str):
        """Thêm thư mục tạm để cleanup sau"""
        if os.path.exists(path):
            self.temp_dirs.append(path)
    
    def add_cache_dir(self, path: str):
        """Thêm thư mục cache để cleanup sau"""
        if os.path.exists(path):
            self.cache_dirs.append(path)
    
    def cleanup_temp_files(self):
        """Xóa các file tạm thời"""
        cleaned_size = 0
        cleaned_count = 0
        
        print("🧹 Dọn dẹp cache tạm thời...")
        
        # Cleanup temp directories
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    size = self._get_dir_size(temp_dir)
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    cleaned_size += size
                    cleaned_count += 1
                    print(f"   ✅ Đã xóa: {temp_dir}")
            except Exception as e:
                print(f"   ⚠️ Không thể xóa {temp_dir}: {e}")
        
        # Cleanup system temp files related to our app
        temp_root = tempfile.gettempdir()
        pixelpure_temps = []
        
        try:
            for item in os.listdir(temp_root):
                if 'pixelpure' in item.lower() or 'clip' in item.lower():
                    full_path = os.path.join(temp_root, item)
                    if os.path.isdir(full_path):
                        pixelpure_temps.append(full_path)
        except:
            pass
        
        for temp_path in pixelpure_temps:
            try:
                size = self._get_dir_size(temp_path)
                shutil.rmtree(temp_path, ignore_errors=True)
                cleaned_size += size
                cleaned_count += 1
                print(f"   ✅ Đã xóa temp: {os.path.basename(temp_path)}")
            except:
                pass
        
        return cleaned_size, cleaned_count
    
    def cleanup_memory_cache(self):
        """Dọn dẹp memory cache"""
        print("🧠 Dọn dẹp memory cache...")
        
        # Clear Python garbage collection
        collected = gc.collect()
        print(f"   ✅ Python GC: {collected} objects")
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("   ✅ CUDA cache cleared")
            except:
                pass
        
        # Clear CPU cache
        try:
            if hasattr(torch, 'cpu'):
                torch.cpu.empty_cache() if hasattr(torch.cpu, 'empty_cache') else None
        except:
            pass
    
    def cleanup_old_logs(self, max_age_days: int = 7):
        """Xóa log files cũ"""
        print("📝 Dọn dẹp log files cũ...")
        
        log_patterns = ['*.log', '*.txt', 'debug_*', 'error_*']
        cleaned_count = 0
        
        for pattern in log_patterns:
            try:
                import glob
                for log_file in glob.glob(pattern):
                    if os.path.isfile(log_file):
                        # Check file age
                        file_age = (os.path.getctime(log_file))
                        import time
                        if time.time() - file_age > (max_age_days * 24 * 3600):
                            os.remove(log_file)
                            cleaned_count += 1
                            print(f"   ✅ Đã xóa log: {log_file}")
            except:
                pass
        
        return cleaned_count
    
    def get_cache_info(self):
        """Lấy thông tin cache hiện tại"""
        info = {
            'model_cache_size': 0,
            'temp_cache_size': 0,
            'total_cache_size': 0
        }
        
        # Model cache size (keep these)
        model_dir = os.path.join("core", "models")
        if os.path.exists(model_dir):
            info['model_cache_size'] = self._get_dir_size(model_dir)
        
        # Temp cache size
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                info['temp_cache_size'] += self._get_dir_size(temp_dir)
        
        info['total_cache_size'] = info['model_cache_size'] + info['temp_cache_size']
        
        return info
    
    def _get_dir_size(self, path: str) -> int:
        """Tính kích thước thư mục"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except:
                        pass
        except:
            pass
        return total_size
    
    def format_size(self, size_bytes: int) -> str:
        """Format kích thước file"""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def full_cleanup(self):
        """Thực hiện cleanup toàn bộ"""
        print("\n🚀 BẮT ĐẦU CLEANUP CACHE")
        print("=" * 50)
        
        # Get initial cache info
        initial_info = self.get_cache_info()
        print(f"📊 Cache ban đầu: {self.format_size(initial_info['total_cache_size'])}")
        print(f"   - Model cache: {self.format_size(initial_info['model_cache_size'])} (giữ lại)")
        print(f"   - Temp cache: {self.format_size(initial_info['temp_cache_size'])} (sẽ xóa)")
        
        # Cleanup operations
        temp_size, temp_count = self.cleanup_temp_files()
        self.cleanup_memory_cache()
        log_count = self.cleanup_old_logs()
        
        # Final info
        final_info = self.get_cache_info()
        saved_space = initial_info['total_cache_size'] - final_info['total_cache_size']
        
        print("\n✅ CLEANUP HOÀN TẤT")
        print(f"🗑️ Đã xóa: {temp_count} thư mục, {log_count} log files")
        print(f"💾 Tiết kiệm: {self.format_size(saved_space)}")
        print(f"📊 Cache còn lại: {self.format_size(final_info['total_cache_size'])}")
        print("=" * 50)

# Global cache manager instance
cache_manager = CacheManager()