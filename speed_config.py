# speed_config.py
# Cấu hình tốc độ quét cho PixelPure

class SpeedConfig:
    """Cấu hình các chế độ quét khác nhau"""
    
    # Chế độ SIÊU NHANH - Ưu tiên tốc độ tối đa
    ULTRA_FAST = {
        'name': 'Siêu Nhanh',
        'description': 'Tốc độ cao nhất, độ chính xác 80-85%',
        'fast_mode': True,
        'batch_size_gpu': 512,
        'batch_size_cpu': 64,
        'model_name': 'ViT-B-32',
        'model_pretrained': 'openai',
        'resize_before_analysis': True,
        'parallel_processing': True,
        'skip_detailed_analysis': True,
        'similarity_threshold': 0.80,
        'max_workers': 8
    }
    
    # Chế độ NHANH - Cân bằng tốc độ và chất lượng
    FAST = {
        'name': 'Nhanh',
        'description': 'Tốc độ cao, độ chính xác 85-90%',
        'fast_mode': True,
        'batch_size_gpu': 256,
        'batch_size_cpu': 32,
        'model_name': 'ViT-B-32',
        'model_pretrained': 'openai',
        'resize_before_analysis': True,
        'parallel_processing': True,
        'skip_detailed_analysis': False,
        'similarity_threshold': 0.82,
        'max_workers': 6
    }
    
    # Chế độ CÂN BẰNG - Mặc định
    BALANCED = {
        'name': 'Cân Bằng',
        'description': 'Cân bằng tốc độ và chất lượng, độ chính xác 90-95%',
        'fast_mode': False,
        'batch_size_gpu': 128,
        'batch_size_cpu': 16,
        'model_name': 'ViT-B-16',
        'model_pretrained': 'openai',
        'resize_before_analysis': False,
        'parallel_processing': True,
        'skip_detailed_analysis': False,
        'similarity_threshold': 0.85,
        'max_workers': 4
    }
    
    # Chế độ CHẤT LƯỢNG CAO - Ưu tiên độ chính xác
    HIGH_QUALITY = {
        'name': 'Chất Lượng Cao',
        'description': 'Độ chính xác cao nhất 95-98%, tốc độ chậm hơn',
        'fast_mode': False,
        'batch_size_gpu': 64,
        'batch_size_cpu': 8,
        'model_name': 'ViT-H-14',
        'model_pretrained': 'laion2b_s32b_b79k',
        'resize_before_analysis': False,
        'parallel_processing': False,
        'skip_detailed_analysis': False,
        'similarity_threshold': 0.87,
        'max_workers': 2
    }
    
    @classmethod
    def get_all_modes(cls):
        """Lấy tất cả các chế độ có sẵn"""
        return {
            'ultra_fast': cls.ULTRA_FAST,
            'fast': cls.FAST,
            'balanced': cls.BALANCED,
            'high_quality': cls.HIGH_QUALITY
        }
    
    @classmethod
    def get_mode(cls, mode_name: str):
        """Lấy cấu hình theo tên chế độ"""
        modes = cls.get_all_modes()
        return modes.get(mode_name, cls.BALANCED)
    
    @classmethod
    def estimate_time(cls, file_count: int, mode_name: str = 'balanced'):
        """Ước tính thời gian xử lý"""
        mode = cls.get_mode(mode_name)
        
        # Ước tính dựa trên số file và chế độ
        base_time_per_file = {
            'ultra_fast': 0.1,    # 0.1 giây/file
            'fast': 0.2,          # 0.2 giây/file  
            'balanced': 0.5,      # 0.5 giây/file
            'high_quality': 1.2   # 1.2 giây/file
        }
        
        time_per_file = base_time_per_file.get(mode_name, 0.5)
        estimated_seconds = file_count * time_per_file
        
        if estimated_seconds < 60:
            return f"~{int(estimated_seconds)} giây"
        elif estimated_seconds < 3600:
            return f"~{int(estimated_seconds/60)} phút"
        else:
            hours = int(estimated_seconds / 3600)
            minutes = int((estimated_seconds % 3600) / 60)
            return f"~{hours}h {minutes}m"