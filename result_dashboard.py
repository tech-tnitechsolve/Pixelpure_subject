# result_dashboard.py
# Dashboard hiển thị kết quả xử lý đẹp mắt

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QGridLayout, QWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class ResultDashboard(QDialog):
    """Dashboard hiển thị kết quả xử lý"""
    
    def __init__(self, result_data, parent=None):
        super().__init__(parent)
        self.result_data = result_data
        self.setWindowTitle("🎉 Kết Quả Xử Lý")
        self.setFixedSize(600, 500)
        self.setModal(True)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        # Header
        header_layout = QVBoxLayout()
        
        title_label = QLabel("🎉 TỰ ĐỘNG XỬ LÝ HOÀN TẤT")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #059669;
            margin-bottom: 10px;
        """)
        
        subtitle_label = QLabel("Workspace của bạn đã được tối ưu hoàn toàn!")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("""
            font-size: 16px;
            color: #6b7280;
            margin-bottom: 20px;
        """)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        layout.addLayout(header_layout)
        
        # Stats cards
        stats_layout = QGridLayout()
        stats_layout.setSpacing(20)
        
        # Card 1: Processed Groups
        processed_card = self.create_stat_card(
            "📊", "Nhóm Đã Xử Lý", 
            str(self.result_data.get('processed_groups', 0)),
            "#3b82f6"
        )
        
        # Card 2: Deleted Files
        deleted_card = self.create_stat_card(
            "🗑️", "File Đã Xóa", 
            str(self.result_data.get('deleted_files', 0)),
            "#ef4444"
        )
        
        # Card 3: Renamed Files
        renamed_card = self.create_stat_card(
            "📝", "File Đã Đổi Tên", 
            str(self.result_data.get('renamed_files', 0)),
            "#f59e0b"
        )
        
        # Card 4: Total Processed
        total_processed = self.result_data.get('deleted_files', 0) + self.result_data.get('renamed_files', 0)
        total_card = self.create_stat_card(
            "✅", "Tổng Xử Lý", 
            str(total_processed),
            "#059669"
        )
        
        stats_layout.addWidget(processed_card, 0, 0)
        stats_layout.addWidget(deleted_card, 0, 1)
        stats_layout.addWidget(renamed_card, 1, 0)
        stats_layout.addWidget(total_card, 1, 1)
        
        layout.addLayout(stats_layout)
        
        # Errors section (if any)
        errors = self.result_data.get('errors', [])
        if errors:
            error_frame = QFrame()
            error_frame.setStyleSheet("""
                QFrame {
                    background-color: #fef2f2;
                    border: 2px solid #fca5a5;
                    border-radius: 8px;
                    padding: 15px;
                }
            """)
            error_layout = QVBoxLayout(error_frame)
            
            error_title = QLabel(f"⚠️ Lỗi ({len(errors)})")
            error_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #dc2626;")
            error_layout.addWidget(error_title)
            
            # Show first 3 errors
            for error in errors[:3]:
                error_label = QLabel(f"• {error}")
                error_label.setStyleSheet("color: #7f1d1d; font-size: 12px;")
                error_label.setWordWrap(True)
                error_layout.addWidget(error_label)
            
            if len(errors) > 3:
                more_label = QLabel(f"... và {len(errors) - 3} lỗi khác")
                more_label.setStyleSheet("color: #7f1d1d; font-size: 12px; font-style: italic;")
                error_layout.addWidget(more_label)
            
            layout.addWidget(error_frame)
        
        # Success message
        if total_processed > 0:
            success_frame = QFrame()
            success_frame.setStyleSheet("""
                QFrame {
                    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                        stop: 0 rgba(5, 150, 105, 0.1), stop: 1 rgba(16, 185, 129, 0.1));
                    border: 2px solid #10b981;
                    border-radius: 8px;
                    padding: 20px;
                }
            """)
            success_layout = QVBoxLayout(success_frame)
            
            success_title = QLabel("🎯 Kết Quả")
            success_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #059669;")
            
            success_text = self.generate_success_message()
            success_label = QLabel(success_text)
            success_label.setStyleSheet("color: #065f46; font-size: 14px; line-height: 1.5;")
            success_label.setWordWrap(True)
            
            success_layout.addWidget(success_title)
            success_layout.addWidget(success_label)
            layout.addWidget(success_frame)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        close_btn = QPushButton("✅ Hoàn Tất")
        close_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #059669, stop: 1 #047857);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #047857, stop: 1 #065f46);
            }
        """)
        close_btn.clicked.connect(self.accept)
        
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        # Overall styling
        self.setStyleSheet("""
            QDialog {
                background-color: #f9fafb;
            }
        """)
    
    def create_stat_card(self, icon, title, value, color):
        """Tạo card thống kê"""
        card = QFrame()
        card.setFixedSize(250, 100)
        card.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 white, stop: 1 rgba(255, 255, 255, 0.8));
                border: 2px solid {color};
                border-radius: 12px;
                padding: 15px;
            }}
            QFrame:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(255, 255, 255, 0.9), stop: 1 rgba(255, 255, 255, 0.7));
            }}
        """)
        
        layout = QHBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"""
            font-size: 32px;
            color: {color};
        """)
        icon_label.setFixedSize(50, 50)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Text
        text_layout = QVBoxLayout()
        text_layout.setSpacing(5)
        
        value_label = QLabel(value)
        value_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {color};
        """)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 12px;
            color: #6b7280;
            font-weight: 500;
        """)
        
        text_layout.addWidget(value_label)
        text_layout.addWidget(title_label)
        
        layout.addWidget(icon_label)
        layout.addLayout(text_layout)
        layout.addStretch()
        
        return card
    
    def generate_success_message(self):
        """Tạo thông điệp thành công"""
        deleted = self.result_data.get('deleted_files', 0)
        renamed = self.result_data.get('renamed_files', 0)
        
        messages = []
        
        if deleted > 0:
            messages.append(f"✅ Đã xóa {deleted} file trùng lặp (giữ lại file tốt nhất)")
        
        if renamed > 0:
            messages.append(f"✅ Đã đổi tên {renamed} file tương tự theo nhóm (1(1), 1(2), 2(1)...)")
        
        if not messages:
            return "Không có file nào cần xử lý."
        
        result = "\n".join(messages)
        result += f"\n\n🎯 Workspace đã được tối ưu hoàn toàn với {deleted + renamed} file được xử lý!"
        
        return result