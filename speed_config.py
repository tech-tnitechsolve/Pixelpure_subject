# speed_config.py
# Cấu hình tốc độ quét cho PixelPure

class SpeedConfig:
    """Cấu hình các chế độ quét khác nhau"""
    
    # --- MODIFICATION: Reverted similarity_threshold to 0.87 for stability ---
    # Chế độ CHẤT LƯỢNG CAO - Ưu tiên độ chính xác
    # Đây là chế độ duy nhất còn lại, được tối ưu cho cả GPU và CPU.
    HIGH_QUALITY = {
        'name': 'Chất Lượng Cao',
        'description': 'Độ chính xác cao nhất 95-98%, tốc độ được tối ưu cho phần cứng.',
        'fast_mode': False,
        'batch_size_gpu': 64,  # Lô xử lý vừa phải cho GPU
        'batch_size_cpu': 4,   # [TỐI ƯU CPU] Giữ ở mức 4 để giảm tải CPU
        'model_name': 'ViT-H-14', # Model AI mạnh nhất
        'model_pretrained': 'laion2b_s32b_b79k',
        'resize_before_analysis': False, # Phân tích trên ảnh gốc
        'parallel_processing': False, # Xử lý tuần tự trên CPU để ổn định
        'skip_detailed_analysis': False,
        'similarity_threshold': 0.87, # [KHÔI PHỤC] Quay về ngưỡng 0.87 ổn định hơn
        'max_workers': 1 # [TỐI ƯU CPU] Giữ ở mức 1 để giảm sử dụng luồng
    }
    
    @classmethod
    def get_all_modes(cls):
        """Lấy tất cả các chế độ có sẵn (chỉ còn lại 1)"""
        return {
            'high_quality': cls.HIGH_QUALITY
        }
    
    @classmethod
    def get_mode(cls, mode_name: str):
        """Lấy cấu hình theo tên chế độ (luôn trả về Chất Lượng Cao)"""
        return cls.HIGH_QUALITY
    
    @classmethod
    def estimate_time(cls, file_count: int, mode_name: str = 'high_quality'):
        """Ước tính thời gian xử lý"""
        time_per_file = 1.2   # Giữ nguyên thời gian ước tính
        estimated_seconds = file_count * time_per_file
        
        if estimated_seconds < 60:
            return f"~{int(estimated_seconds)} giây"
        elif estimated_seconds < 3600:
            return f"~{int(estimated_seconds/60)} phút"
        else:
            hours = int(estimated_seconds / 3600)
            minutes = int((estimated_seconds % 3600) / 60)
            return f"~{hours} giờ {minutes} phút"
