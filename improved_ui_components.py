# improved_ui_components.py
# UI components cải tiến với hiển thị similarity score rõ ràng và nhóm "Hỗn Hợp"

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QCheckBox, QFrame
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap

class ImprovedImageGroupWidget(QWidget):
    """Widget hiển thị nhóm ảnh với UI/UX đơn giản và rõ ràng"""
    delete_clicked = Signal(object)
    move_clicked = Signal(object)

    def __init__(self, files, similarity_score=None, group_type="Unknown", analysis_method=""):
        super().__init__()
        self.files = files
        self.similarity_score = similarity_score
        self.group_type = group_type
        self.analysis_method = analysis_method
        self.checkboxes = []
        self.init_ui()
        
        # [NEW] Default select all for hybrid groups
        if self.group_type == "hybrid_subject":
            self.toggle_all_selection(True)
            self.select_all_cb.setChecked(True)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # Determine colors and text based on type
        if self.group_type == "duplicate":
            type_text = "🔄 TRÙNG LẶP"
            score_text = "100%"
            type_color = "#dc2626"  # Red
            bg_color = "#fef2f2"    # Light red
        elif self.group_type == "similar_subject":
            type_text = "🎯 TƯƠNG TỰ"
            score_val = self.similarity_score if isinstance(self.similarity_score, (int, float)) else 0.85
            score_text = f"{score_val * 100:.0f}%"
            type_color = "#ea580c"  # Orange
            bg_color = "#fff7ed"    # Light orange
        # [NEW] UI for Hybrid Group
        elif self.group_type == "hybrid_subject":
            type_text = "🧬 HỖN HỢP"
            score_val = self.similarity_score if isinstance(self.similarity_score, (int, float)) else 0.90
            score_text = f"~{score_val * 100:.0f}%"
            type_color = "#9333ea"  # Purple
            bg_color = "#f5f3ff"    # Light purple
        else:
            type_text = "❓ KHÔNG XÁC ĐỊNH"
            score_text = "0%"
            type_color = "#6b7280"  # Gray
            bg_color = "#f9fafb"    # Light gray
        
        # Header đơn giản
        header_layout = QHBoxLayout()
        
        # Type và score trong một dòng
        type_score_layout = QHBoxLayout()
        
        type_label = QLabel(type_text)
        type_label.setStyleSheet(f"""
            background-color: {type_color};
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 14px;
        """)
        
        score_label = QLabel(score_text)
        score_label.setStyleSheet(f"""
            color: {type_color};
            font-weight: bold;
            font-size: 24px;
            padding: 6px 12px;
            background-color: {bg_color};
            border-radius: 6px;
            border: 2px solid {type_color};
        """)
        
        type_score_layout.addWidget(type_label)
        type_score_layout.addWidget(score_label)
        type_score_layout.addStretch()
        
        # Info với kích thước chữ lớn hơn
        file_count = len(self.files)
        try:
            total_size = 0
            for f in self.files:
                file_path = f.get('path') if isinstance(f, dict) else f
                if isinstance(file_path, str) and os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
            size_mb = total_size / (1024*1024)
            info_text = f"📁 {file_count} files • 💾 {size_mb:.1f} MB"
        except:
            info_text = f"📁 {file_count} files • 💾 ? MB"
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #4b5563; font-size: 14px; font-weight: 500;")
        
        type_score_layout.addWidget(info_label)
        
        # Actions với kích thước lớn hơn
        actions_layout = QHBoxLayout()
        
        select_all_cb = QCheckBox("Chọn tất cả")
        select_all_cb.setStyleSheet("color: #4b5563; font-size: 13px; font-weight: 500;")
        select_all_cb.clicked.connect(self.toggle_all_selection)
        self.select_all_cb = select_all_cb
        
        delete_btn = QPushButton("🗑️ Xóa")
        delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {type_color};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #b91c1c;
            }}
        """)
        delete_btn.clicked.connect(self.handle_delete)
        
        move_btn = QPushButton("📁 Di chuyển")
        move_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """)
        move_btn.clicked.connect(self.handle_move)
        
        actions_layout.addWidget(select_all_cb)
        actions_layout.addStretch()
        actions_layout.addWidget(delete_btn)
        actions_layout.addWidget(move_btn)
        
        header_layout.addLayout(type_score_layout, 1)
        header_layout.addLayout(actions_layout)
        
        main_layout.addLayout(header_layout)
        
        # Thumbnails với kích thước lớn hơn
        thumbnails_layout = QHBoxLayout()
        thumbnails_layout.setSpacing(12)
        
        # Show max 6 thumbnails
        max_previews = min(6, len(self.files))
        
        for i in range(max_previews):
            file_info = self.files[i]
            file_path = file_info.get('path') if isinstance(file_info, dict) else file_info
            
            if not isinstance(file_path, str) or not os.path.exists(file_path):
                continue
            
            # Thumbnail container lớn hơn
            thumb_container = QWidget()
            thumb_container.setFixedSize(120, 140)
            thumb_layout = QVBoxLayout(thumb_container)
            thumb_layout.setContentsMargins(6, 6, 6, 6)
            thumb_layout.setSpacing(6)
            
            # Checkbox lớn hơn
            checkbox = QCheckBox()
            checkbox.setFixedSize(18, 18)
            self.checkboxes.append(checkbox)
            
            # Thumbnail image lớn hơn
            img_label = QLabel()
            img_label.setFixedSize(106, 90)
            img_label.setStyleSheet(f"""
                border: 2px solid {type_color};
                border-radius: 6px;
                background-color: {bg_color};
            """)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Load thumbnail
            try:
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(102, 86, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    img_label.setPixmap(scaled_pixmap)
                else:
                    img_label.setText("🖼️")
                    img_label.setStyleSheet(img_label.styleSheet() + "font-size: 24px; color: #6b7280;")
            except Exception:
                img_label.setText("❌")
                img_label.setStyleSheet(img_label.styleSheet() + "font-size: 20px; color: #ef4444;")
            
            # File info tooltip
            file_name = os.path.basename(file_path)
            try:
                file_size = os.path.getsize(file_path) / (1024*1024)
                file_info_tooltip = f"{file_name}\n{file_size:.1f} MB\n{file_path}"
            except:
                file_info_tooltip = f"{file_name}\n{file_path}"
            img_label.setToolTip(file_info_tooltip)
            
            # File index lớn hơn
            index_label = QLabel(f"#{i+1}")
            index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            index_label.setFixedHeight(20)
            index_label.setStyleSheet(f"""
                color: {type_color};
                font-size: 12px;
                font-weight: bold;
                background-color: {bg_color};
                border-radius: 10px;
                padding: 2px 6px;
            """)
            
            thumb_layout.addWidget(checkbox, alignment=Qt.AlignmentFlag.AlignCenter)
            thumb_layout.addWidget(img_label)
            thumb_layout.addWidget(index_label)
            
            thumbnails_layout.addWidget(thumb_container)
        
        # "More files" indicator lớn hơn
        if len(self.files) > max_previews:
            more_container = QWidget()
            more_container.setFixedSize(120, 140)
            more_layout = QVBoxLayout(more_container)
            more_layout.setContentsMargins(6, 6, 6, 6)
            
            more_icon = QLabel("⋯")
            more_icon.setFixedSize(106, 90)
            more_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            more_icon.setStyleSheet(f"""
                font-size: 32px;
                color: {type_color};
                border: 2px dashed {type_color};
                border-radius: 6px;
                background-color: {bg_color};
            """)
            
            more_count = QLabel(f"+{len(self.files) - max_previews}")
            more_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
            more_count.setFixedHeight(20)
            more_count.setStyleSheet(f"""
                color: {type_color};
                font-size: 12px;
                font-weight: bold;
                background-color: {bg_color};
                border-radius: 10px;
                padding: 2px 6px;
            """)
            
            more_layout.addWidget(QWidget(), 1)
            more_layout.addWidget(more_icon)
            more_layout.addWidget(more_count)
            more_layout.addWidget(QWidget(), 1)
            
            more_container.setToolTip(f"{len(self.files) - max_previews} more files")
            thumbnails_layout.addWidget(more_container)
        
        thumbnails_layout.addStretch()
        main_layout.addLayout(thumbnails_layout)
        
        # Simple separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: {type_color}; max-height: 1px;")
        main_layout.addWidget(separator)
        
        # Enhanced widget styling
        self.setStyleSheet(f"""
            ImprovedImageGroupWidget {{
                background-color: {bg_color};
                border: 2px solid {type_color};
                border-radius: 10px;
                margin: 12px 0px;
                padding: 4px;
            }}
            ImprovedImageGroupWidget:hover {{
                background-color: white;
                border-color: {type_color};
            }}
        """)

    def toggle_all_selection(self, checked):
        """Toggle all checkboxes"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(checked)

    def get_selected_files(self):
        """Get selected files"""
        selected = []
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked() and i < len(self.files):
                selected.append(self.files[i])
        return selected

    def handle_delete(self):
        """Handle delete button click"""
        selected_files = self.get_selected_files()
        if selected_files:
            self.delete_clicked.emit(selected_files)

    def handle_move(self):
        """Handle move button click"""
        selected_files = self.get_selected_files()
        if selected_files:
            self.move_clicked.emit(selected_files)
