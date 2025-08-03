import sys
import os
import shutil
from typing import Optional, List, Union
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QStackedWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QScrollArea, QCheckBox, QSlider,
    QProgressDialog, QProgressBar, QMessageBox, QFileDialog, QDialog,
    QRadioButton, QButtonGroup, QGridLayout, QFrame
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QCloseEvent

# Safe import for PIL
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

# Safe import for send2trash with fallback
try:
    from send2trash import send2trash as _send2trash
    HAS_SEND2TRASH = True
    def send2trash(path: str) -> None:
        """Wrapper for send2trash with single file support"""
        _send2trash(path)
except ImportError:
    HAS_SEND2TRASH = False
    def send2trash(path: str) -> None:
        """Fallback function when send2trash is not available"""
        os.remove(path)

# Import sau khi đã thiết lập path
try:
    from core.scanner import ScannerWorker
except ImportError:
    ScannerWorker = None

# Dark theme stylesheet
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #ffffff;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}

QLabel {
    color: #ffffff;
    padding: 2px;
}

QPushButton {
    background-color: #404040;
    border: 1px solid #606060;
    border-radius: 6px;
    padding: 8px 16px;
    color: #ffffff;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #505050;
    border-color: #707070;
}

QPushButton:pressed {
    background-color: #303030;
    outline: none;
    border: 1px solid #555555;
}

QPushButton:focus {
    outline: none;
    border: 1px solid #0078d4;
}

QPushButton[cssClass="accent"] {
    background-color: #0078d4;
    border-color: #106ebe;
}

QPushButton[cssClass="accent"]:hover {
    background-color: #106ebe;
}

QPushButton[cssClass="accent"]:pressed {
    background-color: #005a9e;
}

QPushButton[cssClass="danger"] {
    background-color: #d13438;
    border-color: #a1282d;
}

QPushButton[cssClass="danger"]:hover {
    background-color: #a1282d;
}

QTableWidget {
    background-color: #2d2d2d;
    alternate-background-color: #383838;
    gridline-color: #555555;
    border: 1px solid #606060;
    border-radius: 4px;
}

QTableWidget::item {
    padding: 8px;
    border: none;
}

QTableWidget::item:selected {
    background-color: #0078d4;
}

QHeaderView::section {
    background-color: #404040;
    border: 1px solid #606060;
    padding: 8px;
    font-weight: 600;
}

QScrollBar:vertical {
    background-color: #2d2d2d;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #606060;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #707070;
}

QProgressBar {
    border: 1px solid #606060;
    border-radius: 4px;
    text-align: center;
    background-color: #2d2d2d;
}

QProgressBar::chunk {
    background-color: #0078d4;
    border-radius: 3px;
}

QSlider::groove:horizontal {
    border: 1px solid #606060;
    height: 6px;
    background: #2d2d2d;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #0078d4;
    border: 1px solid #106ebe;
    width: 18px;
    margin: -6px 0;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #106ebe;
}

QLabel[status="ok"] {
    color: #4caf50;
    font-weight: 600;
}

QLabel[status="warn"] {
    color: #ff9800;
    font-weight: 600;
}

QLabel[status="error"] {
    color: #f44336;
    font-weight: 600;
}

QGroupBox {
    font-weight: 600;
    border: 2px solid #606060;
    border-radius: 8px;
    margin: 4px 0px;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
}

QCheckBox {
    spacing: 5px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
}

QCheckBox::indicator:unchecked {
    border: 2px solid #606060;
    background-color: #2d2d2d;
    border-radius: 3px;
}

QCheckBox::indicator:checked {
    border: 2px solid #0078d4;
    background-color: #0078d4;
    border-radius: 3px;
}

QCheckBox::indicator:checked:hover {
    background-color: #106ebe;
}

QMessageBox {
    background-color: #1e1e1e;
}

QDialog {
    background-color: #1e1e1e;
}

QRadioButton {
    spacing: 5px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
}

QRadioButton::indicator:unchecked {
    border: 2px solid #606060;
    background-color: #2d2d2d;
    border-radius: 8px;
}

QRadioButton::indicator:checked {
    border: 2px solid #0078d4;
    background-color: #0078d4;
    border-radius: 8px;
}
"""

class PixelPureApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setApplicationName("PixelPure")
        self.setApplicationVersion("1.0")
        self.main_window: Optional['MainWindow'] = None

class DropZone(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        icon_label = QLabel("📁")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 64px;")

        title_label = QLabel("PixelPure - Dọn dẹp thông minh")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")

        instruction_label = QLabel("Kéo thả file ảnh hoặc thư mục vào đây\nHoặc nhấn nút bên dưới để chọn file")
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instruction_label.setStyleSheet("font-size: 14px; color: #cccccc; line-height: 1.5;")

        button_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("Chọn File Ảnh")
        self.add_folder_btn = QPushButton("Chọn Thư Mục")
        
        self.add_files_btn.setProperty("cssClass", "accent")
        self.add_folder_btn.setProperty("cssClass", "accent")
        
        button_layout.addWidget(self.add_files_btn)
        button_layout.addWidget(self.add_folder_btn)

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(instruction_label)
        layout.addLayout(button_layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        # Find the MainWindow through the widget hierarchy
        widget = self
        while widget and not isinstance(widget, MainWindow):
            widget = widget.parent()
        if widget and hasattr(widget, 'handle_dropped_files'):
            widget.handle_dropped_files(paths)

class FileTableWidget(QTableWidget):
    def __init__(self):
        super().__init__()
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)

class CompactImageGroupWidget(QWidget):
    delete_clicked = Signal(object)
    move_clicked = Signal(object)

    def __init__(self, files, similarity_score=None, group_type="Unknown"):
        super().__init__()
        self.files = files
        self.similarity_score = similarity_score
        self.group_type = group_type
        self.checkboxes = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(20)
        
        # Column 1: Compact Status Info (120px width)
        status_widget = QWidget()
        status_widget.setFixedWidth(120)
        status_layout = QVBoxLayout(status_widget)
        status_layout.setContentsMargins(10, 8, 10, 8)
        status_layout.setSpacing(6)
        
        # Simple type indicator
        if self.group_type == "Duplicate":
            type_text = "TRÙNG LẶP"
            score_text = "100%"
            type_color = "#ef4444"
        elif self.group_type == "Subject":
            type_text = "TƯƠNG TỰ"
            # Fix similarity score calculation - get actual score
            if isinstance(self.similarity_score, (int, float)):
                score_text = f"{self.similarity_score:.0f}%"
            else:
                score_text = "85%"  # Default fallback
            type_color = "#f59e0b"
        else:
            type_text = "KHÔNG XÁC ĐỊNH"
            score_text = "0%"
            type_color = "#6b7280"
        
        # Clean type badge
        type_label = QLabel(type_text)
        type_label.setStyleSheet(f"""
            background-color: {type_color};
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 9px;
        """)
        type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Large similarity score
        score_label = QLabel(score_text)
        score_label.setStyleSheet(f"""
            color: {type_color};
            font-weight: bold;
            font-size: 18px;
        """)
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # File count
        file_count = len(self.files)
        count_label = QLabel(f"{file_count} files")
        count_label.setStyleSheet("color: #9ca3af; font-size: 11px;")
        count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        status_layout.addWidget(type_label)
        status_layout.addWidget(score_label)
        status_layout.addWidget(count_label)
        status_layout.addStretch()
        
        # Column 2: Larger Thumbnails (main content area)
        preview_widget = QWidget()
        preview_layout = QHBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(12)
        
        # Show max 6 larger thumbnails
        max_previews = min(6, len(self.files))
        
        for i in range(max_previews):
            file_path = self.files[i]
            if not isinstance(file_path, str) or not os.path.exists(file_path):
                continue
            
            # Larger thumbnail container (increased by 1.75x: 90 -> 158, 110 -> 193)
            thumb_container = QWidget()
            thumb_container.setFixedSize(158, 193)
            thumb_layout = QVBoxLayout(thumb_container)
            thumb_layout.setContentsMargins(6, 6, 6, 6)
            thumb_layout.setSpacing(6)
            
            # Checkbox at top
            checkbox = QCheckBox()
            checkbox.setFixedSize(20, 20)
            self.checkboxes.append(checkbox)
            
            # Much larger thumbnail image (82 -> 144)
            img_label = QLabel()
            img_label.setFixedSize(144, 144)
            img_label.setStyleSheet("""
                border: 2px solid #374151;
                border-radius: 8px;
                background-color: #1f2937;
            """)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Load thumbnail with better size
            try:
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(140, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    img_label.setPixmap(scaled_pixmap)
                else:
                    img_label.setText("🖼️")
                    img_label.setStyleSheet(img_label.styleSheet() + "font-size: 32px; color: #6b7280;")
            except Exception:
                img_label.setText("❌")
                img_label.setStyleSheet(img_label.styleSheet() + "font-size: 28px; color: #ef4444;")
            
            # Simple tooltip with file info
            file_name = os.path.basename(file_path)
            try:
                file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                file_info = f"📁 {file_name}\n📊 {file_size:.1f} MB\n📍 {file_path}"
            except:
                file_info = f"📁 {file_name}\n📍 {file_path}"
            img_label.setToolTip(file_info)
            
            # Hover effect for thumbnail
            img_label.setStyleSheet(img_label.styleSheet() + """
                QLabel:hover {
                    border: 2px solid #3b82f6;
                    background-color: rgba(59, 130, 246, 0.1);
                }
            """)
            
            # File index
            index_label = QLabel(f"{i+1}")
            index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            index_label.setFixedHeight(16)
            index_label.setStyleSheet("""
                color: #9ca3af;
                font-size: 10px;
                font-weight: bold;
            """)
            
            thumb_layout.addWidget(checkbox, alignment=Qt.AlignmentFlag.AlignCenter)
            thumb_layout.addWidget(img_label)
            thumb_layout.addWidget(index_label)
            
            # Hover effect for container
            thumb_container.setStyleSheet("""
                QWidget {
                    border-radius: 4px;
                }
                QWidget:hover {
                    background-color: rgba(55, 65, 81, 0.3);
                }
            """)
            
            preview_layout.addWidget(thumb_container)
        
        # "More files" indicator if needed
        if len(self.files) > max_previews:
            more_container = QWidget()
            more_container.setFixedSize(90, 110)
            more_layout = QVBoxLayout(more_container)
            more_layout.setContentsMargins(4, 4, 4, 4)
            
            more_icon = QLabel("···")
            more_icon.setFixedSize(82, 82)
            more_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            more_icon.setStyleSheet("""
                font-size: 32px;
                color: #6b7280;
                border: 2px dashed #374151;
                border-radius: 6px;
                background-color: #1f2937;
            """)
            
            more_count = QLabel(f"+{len(self.files) - max_previews}")
            more_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
            more_count.setStyleSheet("color: #9ca3af; font-size: 10px; font-weight: bold;")
            
            more_layout.addWidget(QWidget(), 1)  # Spacer
            more_layout.addWidget(more_icon)
            more_layout.addWidget(more_count)
            more_layout.addWidget(QWidget(), 1)  # Spacer
            
            more_container.setToolTip(f"{len(self.files) - max_previews} more files")
            preview_layout.addWidget(more_container)
        
        preview_layout.addStretch()
        
        # Column 3: Simplified Actions (80px width)
        actions_widget = QWidget()
        actions_widget.setFixedWidth(80)
        actions_layout = QVBoxLayout(actions_widget)
        actions_layout.setContentsMargins(5, 5, 5, 5)
        actions_layout.setSpacing(8)
        
        # Simple action buttons
        delete_btn = QPushButton("🗑️")
        delete_btn.setFixedSize(30, 30)
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                border: none;
                border-radius: 15px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b91c1c;
            }
        """)
        delete_btn.setToolTip("Delete selected files")
        delete_btn.clicked.connect(self.handle_delete)
        
        move_btn = QPushButton("�")
        move_btn.setFixedSize(30, 30)
        move_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 15px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """)
        move_btn.setToolTip("Move selected files")
        move_btn.clicked.connect(self.handle_move)
        
        # Select all checkbox
        select_all_cb = QCheckBox("All")
        select_all_cb.setStyleSheet("color: #9ca3af; font-size: 10px;")
        select_all_cb.clicked.connect(self.toggle_all_selection)
        
        actions_layout.addWidget(select_all_cb, alignment=Qt.AlignmentFlag.AlignCenter)
        actions_layout.addSpacing(5)
        actions_layout.addWidget(delete_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        actions_layout.addWidget(move_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        actions_layout.addStretch()
        
        # Store select all checkbox reference
        self.select_all_cb = select_all_cb
        
        # Add all columns
        main_layout.addWidget(status_widget)
        main_layout.addWidget(preview_widget, 1)  # Stretch factor
        main_layout.addWidget(actions_widget)
        
        # Clean widget styling with better separation
        self.setStyleSheet("""
            CompactImageGroupWidget {
                background-color: #111827;
                border: 1px solid #374151;
                border-radius: 8px;
                margin: 8px 0px;
            }
            CompactImageGroupWidget:hover {
                border-color: #3b82f6;
                background-color: #1f2937;
            }
        """)

    def toggle_all_selection(self, checked):
        """Toggle all checkboxes based on select all checkbox"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(checked)

    def get_selected_files(self):
        selected = []
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked() and i < len(self.files):
                selected.append(self.files[i])
        return selected

    def handle_delete(self):
        selected_files = self.get_selected_files()
        if selected_files:
            self.delete_clicked.emit(selected_files)

    def handle_move(self):
        selected_files = self.get_selected_files()
        if selected_files:
            self.move_clicked.emit(selected_files)




class ImageGroupWidget(QWidget):
    delete_clicked = Signal(object)
    move_clicked = Signal(object)

    def __init__(self, files, similarity_score=None, group_type="Unknown"):
        super().__init__()
        self.files = files
        self.similarity_score = similarity_score
        self.group_type = group_type
        self.checkboxes = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header với thông tin nhóm
        header_layout = QHBoxLayout()
        
        if self.group_type == "Duplicate":
            type_icon = "🔄"
            type_text = "TRÙNG LẶP 100%"
            type_color = "#f44336"
        elif self.group_type == "Subject":
            type_icon = "🎯" 
            type_text = f"SUBJECT {self.similarity_score:.1f}%"
            type_color = "#ff9800"
        else:
            type_icon = "❓"
            type_text = "UNKNOWN"
            type_color = "#9e9e9e"
        
        type_label = QLabel(f"{type_icon} {type_text}")
        type_label.setStyleSheet(f"font-weight: bold; color: {type_color}; font-size: 12px;")
        
        # Safe file size calculation
        try:
            total_size = 0
            file_count = len(self.files)
            for f in self.files:
                if isinstance(f, str) and os.path.exists(f):
                    try:
                        total_size += os.path.getsize(f)
                    except (OSError, TypeError):
                        continue
            size_mb = total_size / (1024*1024)
            info_label = QLabel(f"📊 {file_count} file • {size_mb:.1f} MB")
        except Exception:
            info_label = QLabel(f"📊 {len(self.files)} file • ? MB")
        info_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        
        header_layout.addWidget(type_label)
        header_layout.addStretch()
        header_layout.addWidget(info_label)
        
        layout.addLayout(header_layout)

        # Grid layout cho ảnh
        images_widget = QWidget()
        images_layout = QGridLayout(images_widget)
        images_layout.setSpacing(5)
        
        cols = min(4, len(self.files))
        
        for i, file_path in enumerate(self.files):
            # Ensure file_path is string
            if not isinstance(file_path, str):
                continue
            if not os.path.exists(file_path):
                continue
                
            row, col = divmod(i, cols)
            
            # Container cho mỗi ảnh
            img_container = QWidget()
            img_layout = QVBoxLayout(img_container)
            img_layout.setContentsMargins(2, 2, 2, 2)
            img_layout.setSpacing(2)
            
            # Checkbox
            checkbox = QCheckBox()
            self.checkboxes.append(checkbox)
            
            # Label ảnh
            img_label = QLabel()
            img_label.setFixedSize(120, 120)
            img_label.setStyleSheet("border: 1px solid #606060; border-radius: 4px;")
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            try:
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(118, 118, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    img_label.setPixmap(scaled_pixmap)
                else:
                    img_label.setText("❌")
            except Exception:
                img_label.setText("❌")
            
            # Tên file (rút gọn)
            filename = os.path.basename(file_path)
            if len(filename) > 15:
                filename = filename[:12] + "..."
            name_label = QLabel(filename)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("font-size: 9px; color: #cccccc;")
            
            img_layout.addWidget(checkbox)
            img_layout.addWidget(img_label)
            img_layout.addWidget(name_label)
            
            images_layout.addWidget(img_container, row, col)
        
        layout.addWidget(images_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Chọn tất cả")
        select_all_btn.clicked.connect(self.select_all)
        
        deselect_all_btn = QPushButton("Bỏ chọn tất cả") 
        deselect_all_btn.clicked.connect(self.deselect_all)
        
        delete_btn = QPushButton("🗑️ Xóa đã chọn")
        delete_btn.setProperty("cssClass", "danger")
        delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self))
        
        move_btn = QPushButton("📁 Di chuyển nhóm")
        move_btn.clicked.connect(lambda: self.move_clicked.emit(self))
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        button_layout.addStretch()
        button_layout.addWidget(delete_btn)
        button_layout.addWidget(move_btn)
        
        layout.addLayout(button_layout)
        
        # Styling
        self.setStyleSheet("""
        ImageGroupWidget {
            border: 1px solid #606060;
            border-radius: 8px;
            background-color: #2d2d2d;
            margin: 5px;
            padding: 5px;
        }
        """)

    def select_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)

    def deselect_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)

    def get_selected_files(self):
        selected = []
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked() and i < len(self.files):
                file_path = self.files[i]
                if isinstance(file_path, str):
                    selected.append(file_path)
        return selected

    def get_all_files(self):
        # Ensure all files are strings
        return [f for f in self.files if isinstance(f, str)]

class ResultsView(QWidget):
    def __init__(self):
        super().__init__()
        self.group_widgets = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("🔍 KẾT QUẢ QUÉT")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        self.scan_again_btn = QPushButton("🔄 Quét lại")
        self.scan_again_btn.setProperty("cssClass", "accent")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.scan_again_btn)
        
        layout.addLayout(header_layout)

        # Modern Summary và Execute All với enhanced design
        summary_layout = QHBoxLayout()
        summary_layout.setContentsMargins(8, 12, 8, 12)
        
        # Enhanced summary label with modern styling
        self.results_summary_label = QLabel("📊 Đang tải kết quả...")
        self.results_summary_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(56, 161, 105, 0.2), stop: 1 rgba(49, 130, 206, 0.2));
                color: #e2e8f0;
                font-weight: 600;
                font-size: 13px;
                padding: 12px 20px;
                border-radius: 8px;
                border: 1px solid rgba(56, 161, 105, 0.3);
                font-family: 'Segoe UI', system-ui;
            }
        """)
        
        # Enhanced Execute All button with modern design
        self.execute_all_btn = QPushButton("⚡ TỰ ĐỘNG XỬ LÝ")
        self.execute_all_btn.setFixedHeight(45)
        self.execute_all_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #059669, stop: 0.5 #047857, stop: 1 #065f46);
                color: white;
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 0.5px;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                font-family: 'Segoe UI', system-ui;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #047857, stop: 0.5 #065f46, stop: 1 #064e3b);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #064e3b, stop: 1 #047857);
            }
        """)
        self.execute_all_btn.clicked.connect(self.execute_all_changes)
        
        summary_layout.addWidget(self.results_summary_label, 1)  # Stretch factor
        summary_layout.addSpacing(15)
        summary_layout.addWidget(self.execute_all_btn)
        
        layout.addLayout(summary_layout)

        # Info panel
        info_layout = QHBoxLayout()
        
        threshold_info = QLabel(
            f"🧠 Subject Analysis V25+ NGHIÊM NGẶT (85% threshold)"
        )
        threshold_info.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 11px;")
        
        info_layout.addWidget(threshold_info)
        info_layout.addStretch()
        
        layout.addLayout(info_layout)

        # Enhanced scroll area với modern styling
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Modern scroll area styling
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid rgba(74, 85, 104, 0.4);
                border-radius: 8px;
                background-color: rgba(26, 32, 44, 0.3);
            }
            QScrollBar:vertical {
                background-color: rgba(45, 55, 72, 0.5);
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(99, 179, 237, 0.6);
                border-radius: 6px;
                min-height: 20px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: rgba(99, 179, 237, 0.8);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: rgba(45, 55, 72, 0.5);
                height: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: rgba(99, 179, 237, 0.6);
                border-radius: 6px;
                min-width: 20px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: rgba(99, 179, 237, 0.8);
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
                width: 0px;
            }
        """)
        
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setSpacing(12)  # Better spacing between groups
        self.scroll_layout.setContentsMargins(12, 12, 12, 12)
        
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)

    def populate_results(self, results):
        # Clear existing widgets
        for widget in self.group_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self.group_widgets.clear()

        # Phân loại kết quả
        duplicate_count = 0
        subject_count = 0
        total_files = 0

        for i, group in enumerate(results):
            group_type = group.get('type', 'Unknown')
            similarity = group.get('similarity', 0)
            files = group.get('files', [])
            
            # Ensure files is a list and extract paths
            if not isinstance(files, list):
                continue
            
            # Extract file paths from file objects
            original_files = files[:]
            file_paths = []
            for file_obj in files:
                if isinstance(file_obj, dict) and 'path' in file_obj:
                    file_paths.append(file_obj['path'])
                elif isinstance(file_obj, str):
                    file_paths.append(file_obj)
            
            if not file_paths:
                continue
            
            files = file_paths  # Use extracted paths
            
            total_files += len(files)
            
            if group_type == 'duplicate':
                duplicate_count += 1
                group_widget = CompactImageGroupWidget(files, 100.0, "Duplicate")
            else:
                subject_count += 1
                group_widget = CompactImageGroupWidget(files, similarity, "Subject")
            
            group_widget.delete_clicked.connect(self.handle_delete)
            group_widget.move_clicked.connect(self.handle_move)

            # Add separator between groups (except for first group)
            if len(self.group_widgets) > 0:
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.HLine)
                separator.setFrameShadow(QFrame.Shadow.Sunken)
                separator.setStyleSheet("""
                    QFrame {
                        color: #4b5563;
                        background-color: #4b5563;
                        border: none;
                        height: 1px;
                        margin: 12px 20px;
                    }
                """)
                self.scroll_layout.addWidget(separator)
            
            self.group_widgets.append(group_widget)
            self.scroll_layout.addWidget(group_widget)

        # Simple header separator only if results exist
        if results:
            separator = QWidget()
            separator.setFixedHeight(1)
            separator.setStyleSheet("background-color: rgba(55, 65, 81, 0.5); margin: 8px 0px;")
            self.scroll_layout.addWidget(separator)

        # Cập nhật summary với enhanced formatting
        total_groups = len(results)
        
        # Create detailed statistics with visual indicators
        summary_parts = []
        if total_groups > 0:
            summary_parts.append(f"📊 <b>{total_groups}</b> groups detected")
        if duplicate_count > 0:
            summary_parts.append(f"🔄 <b style='color:#e53e3e'>{duplicate_count}</b> duplicates")
        if subject_count > 0:
            summary_parts.append(f"🎯 <b style='color:#dd6b20'>{subject_count}</b> similar")
        if total_files > 0:
            summary_parts.append(f"📁 <b style='color:#3182ce'>{total_files}</b> files")
        
        summary_text = " • ".join(summary_parts) if summary_parts else "📊 No results to display"
        
        # Calculate potential space savings
        if duplicate_count > 0:
            potential_savings = f" • 💾 <span style='color:#38a169'>~{duplicate_count * 20:.0f}MB</span> can be saved"
            summary_text += potential_savings
        
        self.results_summary_label.setText(summary_text)

        # Add stretch at the end
        self.scroll_layout.addStretch()

    def execute_all_changes(self):
        """Tự động triển khai: Xóa duplicates + Đổi tên similar files"""
        if not self.group_widgets:
            QMessageBox.information(self, "Không có dữ liệu", "Không có nhóm nào để xử lý.")
            return

        # Phân loại nhóm
        duplicate_groups = []
        similar_groups = []
        
        for group_widget in self.group_widgets:
            if group_widget.group_type == "Duplicate":
                duplicate_groups.append(group_widget)
            elif group_widget.group_type == "Subject":
                similar_groups.append(group_widget)

        total_files = sum(len(g.files) for g in self.group_widgets)
        duplicate_files = sum(len(g.files) - 1 for g in duplicate_groups)  # Trừ 1 file giữ lại
        similar_files = sum(len(g.files) for g in similar_groups)
        
        # Hiển thị thông tin tổng quan ngắn gọn
        summary_text = f"""
⚡ TỰ ĐỘNG TRIỂN KHAI TOÀN BỘ:

📊 THỐNG KÊ:
• {len(duplicate_groups)} nhóm trùng lặp → Xóa {duplicate_files} file (giữ lại 1/nhóm)
• {len(similar_groups)} nhóm tương tự → Đổi tên {similar_files} file theo format nhóm
• Tổng xử lý: {total_files} file

🚀 TỰ ĐỘNG HOÀN TOÀN - KHÔNG CẦN THAO TÁC THÊM!
        """

        reply = QMessageBox.question(
            self, "⚡ TỰ ĐỘNG TRIỂN KHAI", 
            summary_text + "\n\nBạn có muốn tiếp tục?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Bước 1: Tự động xóa các file duplicate
        if duplicate_groups:
            self._auto_delete_duplicates(duplicate_groups)

        # Bước 2: Tự động đổi tên các file similar
        if similar_groups:
            self._auto_rename_similar_files(similar_groups)

        # Thông báo hoàn tất
        QMessageBox.information(
            self, "🎉 HOÀN TẤT TỰ ĐỘNG!", 
            f"✅ Đã xử lý xong {total_files} file!\n\n"
            f"🗑️ Duplicates: {duplicate_files} file → thùng rác\n"
            f"🔤 Similar: {similar_files} file → đổi tên nhóm\n\n"
            f"📁 Workspace đã được tối ưu hoàn toàn!"
        )

        # Clear tất cả groups
        for group in self.group_widgets:
            group.setParent(None)
            group.deleteLater()
        self.group_widgets.clear()
        self.results_summary_label.setText("🎯 Đã xử lý xong tất cả - Workspace sạch sẽ!")

    def _auto_delete_duplicates(self, duplicate_groups):
        """Tự động xóa tất cả file duplicates"""
        files_to_delete = []
        
        for group in duplicate_groups:
            # Chọn tất cả file trừ file đầu tiên (giữ lại 1 file)
            if len(group.files) > 1:
                files_to_delete.extend(group.files[1:])  # Bỏ qua file đầu tiên
        
        if not files_to_delete:
            return

        progress = QProgressDialog(
            "Đang xóa file trùng lặp...", "Hủy", 0, len(files_to_delete), self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("🗑️ Đang xóa duplicates...")
        progress.show()
        
        deleted_count, errors = 0, []
        
        for i, file_path in enumerate(files_to_delete):
            if progress.wasCanceled():
                break
                
            progress.setValue(i)
            progress.setLabelText(f"Đang xóa: {os.path.basename(file_path)}")
            QApplication.processEvents()
            
            try:
                if HAS_SEND2TRASH:
                    send2trash(file_path)
                else:
                    os.remove(file_path)
                deleted_count += 1
            except Exception as e: 
                errors.append(f"❌ {os.path.basename(file_path)}: {str(e)}")
        
        progress.setValue(len(files_to_delete))
        progress.close()
        
        # Thông báo kết quả
        if errors:
            QMessageBox.warning(
                self, "⚠️ Hoàn tất với lỗi", 
                f"✅ Đã xóa: {deleted_count}/{len(files_to_delete)} file duplicate\n"
                f"❌ Lỗi: {len(errors)} file\n\n"
                f"Chi tiết lỗi:\n{chr(10).join(errors[:5])}"
                + ("..." if len(errors) > 5 else "")
            )
        else:
            action_text = "chuyển vào thùng rác" if HAS_SEND2TRASH else "xóa vĩnh viễn"
            QMessageBox.information(
                self, "🎉 Thành công!", 
                f"✅ Đã {action_text} {deleted_count} file duplicate!\n"
                f"🔄 Giữ lại 1 file đại diện cho mỗi nhóm."
            )
        
        # Remove duplicate groups from UI
        for group in duplicate_groups:
            group.setParent(None)
            group.deleteLater()
            if group in self.group_widgets:
                self.group_widgets.remove(group)

    def _auto_rename_similar_files(self, similar_groups):
        """Tự động đổi tên similar files theo nhóm"""
        total_files = sum(len(group.files) for group in similar_groups)
        progress = QProgressDialog(
            "Đang đổi tên file tương tự...", "Hủy", 0, total_files, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("🔤 Đang đổi tên tự động...")
        progress.show()
        
        renamed_count, errors = 0, []
        file_index = 0
        
        for group_index, group in enumerate(similar_groups, 1):
            for file_index_in_group, file_path in enumerate(group.files, 1):
                if progress.wasCanceled():
                    break
                    
                file_index += 1
                progress.setValue(file_index)
                old_filename = os.path.basename(file_path)
                progress.setLabelText(f"Nhóm {group_index}: {old_filename}")
                QApplication.processEvents()
                
                try:
                    # Tạo tên mới theo format: GroupNumber(FileNumber).extension
                    file_dir = os.path.dirname(file_path)
                    _, ext = os.path.splitext(file_path)
                    new_filename = f"{group_index}({file_index_in_group}){ext}"
                    new_path = os.path.join(file_dir, new_filename)
                    
                    # Xử lý trùng tên
                    counter = 1
                    while os.path.exists(new_path):
                        new_filename = f"{group_index}({file_index_in_group})_{counter}{ext}"
                        new_path = os.path.join(file_dir, new_filename)
                        counter += 1
                    
                    os.rename(file_path, new_path)
                    renamed_count += 1
                except Exception as e:
                    errors.append(f"❌ {old_filename}: {str(e)}")
        
        progress.setValue(total_files)
        progress.close()
        
        # Remove groups from UI
        for group in similar_groups:
            group.setParent(None)
            group.deleteLater()
            if group in self.group_widgets:
                self.group_widgets.remove(group)

    def handle_delete(self, group_widget):
        files = group_widget.get_selected_files()
        if not files:
            QMessageBox.warning(self, "Chưa chọn file", "Vui lòng chọn ít nhất một file để xóa.")
            return
        reply = QMessageBox.question(self, "Xác nhận xóa", f"Bạn có chắc muốn xóa vĩnh viễn {len(files)} file?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            deleted_count, errors = 0, []
            for f in files:
                try: os.remove(f); deleted_count += 1
                except Exception as e: errors.append(f"{os.path.basename(f)}: {e}")
            QMessageBox.information(self, "Hoàn tất", f"Đã xóa thành công {deleted_count} file.")
            if errors: QMessageBox.critical(self, "Lỗi", "Không thể xóa một số file:\n" + "\n".join(errors))
            
            group_widget.setParent(None)
            group_widget.deleteLater()
            self.group_widgets.remove(group_widget)
            self.results_summary_label.setText(f"Còn lại {len(self.group_widgets)} nhóm.")

    def handle_move(self, group_widget):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục để di chuyển nhóm ảnh")
        if not folder: 
            return
        files = group_widget.get_all_files()
        i = 1
        while True:
            subfolder_name = f"nhom_{i}"
            dest_folder = os.path.join(folder, subfolder_name)
            if not os.path.exists(dest_folder): 
                os.makedirs(dest_folder)
                break
            i += 1
        moved_count, errors = 0, []
        for f_path in files:
            try: 
                shutil.move(f_path, os.path.join(dest_folder, os.path.basename(f_path)))
                moved_count += 1
            except Exception as e: 
                errors.append(f"{os.path.basename(f_path)}: {e}")
        QMessageBox.information(self, "Hoàn tất", f"Đã di chuyển {moved_count} file vào thư mục '{subfolder_name}'.")
        if errors: 
            QMessageBox.critical(self, "Lỗi", "Không thể di chuyển một số file:\n" + "\n".join(errors))
        
        group_widget.setParent(None)
        group_widget.deleteLater()
        self.group_widgets.remove(group_widget)
        self.results_summary_label.setText(f"Còn lại {len(self.group_widgets)} nhóm.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixelPure - Dọn dẹp thông minh, Tối ưu không gian")
        self.setGeometry(100, 100, 1200, 800)
        self.file_list = set()
        
        self.scanner_worker = None
        
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.create_drop_zone_view()
        self.create_file_list_view()
        self.results_view = ResultsView()
        self.stacked_widget.addWidget(self.results_view)
        self.results_view.scan_again_btn.clicked.connect(self.reset_app)
        self.stacked_widget.setCurrentIndex(0)

    def create_drop_zone_view(self):
        drop_zone = DropZone()
        drop_zone.add_files_btn.clicked.connect(self.add_files)
        drop_zone.add_folder_btn.clicked.connect(self.add_folder)
        self.stacked_widget.addWidget(drop_zone)

    def create_file_list_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Top bar
        top_bar = QHBoxLayout()
        self.add_file_btn = QPushButton("Thêm File")
        self.add_folder_btn = QPushButton("Thêm Thư Mục")
        self.select_all_btn = QPushButton("Chọn Tất Cả")
        self.deselect_all_btn = QPushButton("Bỏ Chọn Tất Cả")
        
        top_bar.addWidget(self.add_file_btn)
        top_bar.addWidget(self.add_folder_btn)
        top_bar.addStretch()
        top_bar.addWidget(self.select_all_btn)
        top_bar.addWidget(self.deselect_all_btn)

        # File table
        self.file_table = FileTableWidget()
        self.file_table.setColumnCount(5)
        self.file_table.setHorizontalHeaderLabels(["", "Tên File", "Kích Thước", "Định Dạng", "Đường Dẫn"])
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.file_table.setColumnWidth(0, 30)
        
        # Bottom bar
        bottom_bar = QHBoxLayout()
        self.stats_label = QLabel("Tổng số ảnh: 0 | Tổng dung lượng: 0 MB")
        
        self.device_status_label = QLabel()
        self.device_status_label.setObjectName("gpu_status")
        self.update_device_status()

        # Threshold slider (fixed at 85%)
        slider_layout = QHBoxLayout()
        slider_label = QLabel("Ngưỡng Subject (Cố định):")
        self.similarity_slider = QSlider(Qt.Orientation.Horizontal)
        self.similarity_slider.setMinimum(80)
        self.similarity_slider.setMaximum(95)
        self.similarity_slider.setValue(85)
        self.similarity_slider.setEnabled(False)  # Disabled because it's fixed
        self.slider_value_label = QLabel(f"{self.similarity_slider.value()}% NGHIÊM NGẶT")
        
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.similarity_slider)
        slider_layout.addWidget(self.slider_value_label)
        
        self.reset_btn = QPushButton("Reset")
        self.start_scan_btn = QPushButton("Bắt Đầu Quét")
        self.start_scan_btn.setProperty("cssClass", "accent")
        
        bottom_bar.addWidget(self.stats_label)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.device_status_label)
        bottom_bar.addSpacing(20)
        bottom_bar.addLayout(slider_layout)
        bottom_bar.addSpacing(20)
        bottom_bar.addWidget(self.reset_btn)
        bottom_bar.addWidget(self.start_scan_btn)
        
        layout.addLayout(top_bar)
        layout.addWidget(self.file_table)
        layout.addLayout(bottom_bar)
        
        self.stacked_widget.addWidget(widget)

        # Connect signals
        self.add_file_btn.clicked.connect(self.add_files)
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.select_all_btn.clicked.connect(self.select_all)
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        self.reset_btn.clicked.connect(self.reset_app)
        self.start_scan_btn.clicked.connect(self.start_scan)
        self.similarity_slider.valueChanged.connect(lambda v: self.slider_value_label.setText(f"{v}% NGHIÊM NGẶT"))

    def update_device_status(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.device_status_label.setText(f"DEVICE: {torch.cuda.get_device_name(0)}")
                self.device_status_label.setProperty("status", "ok")
                self.device_status_label.setToolTip("Ứng dụng đang sử dụng GPU để tăng tốc xử lý.")
            else:
                self.device_status_label.setText("DEVICE: CPU")
                self.device_status_label.setProperty("status", "warn")
                self.device_status_label.setToolTip("Không tìm thấy GPU tương thích. Ứng dụng sẽ chạy trên CPU và có thể chậm hơn.")
        except ImportError:
            self.device_status_label.setText("DEVICE: CPU")
            self.device_status_label.setProperty("status", "warn")
            self.device_status_label.setToolTip("PyTorch chưa được cài đặt.")
        
        self.device_status_label.style().unpolish(self.device_status_label)
        self.device_status_label.style().polish(self.device_status_label)

    def handle_dropped_files(self, paths):
        valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        added_count = 0
        
        for path in paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for name in files:
                        if os.path.splitext(name)[1].lower() in valid_ext:
                            full_path = os.path.join(root, name)
                            if full_path not in self.file_list:
                                self.file_list.add(full_path)
                                added_count += 1
            elif os.path.isfile(path) and os.path.splitext(path)[1].lower() in valid_ext:
                if path not in self.file_list:
                    self.file_list.add(path)
                    added_count += 1
        
        if added_count > 0:
            self.update_file_table()
        
        if self.file_list:
            self.stacked_widget.setCurrentIndex(1)

    def update_file_table(self):
        self.file_table.setRowCount(0)
        total_size = 0
        
        for file_path in sorted(list(self.file_list)):
            try:
                row = self.file_table.rowCount()
                self.file_table.insertRow(row)
                
                chk = QTableWidgetItem()
                chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                chk.setCheckState(Qt.CheckState.Checked)
                self.file_table.setItem(row, 0, chk)
                
                size = os.path.getsize(file_path)
                total_size += size
                
                self.file_table.setItem(row, 1, QTableWidgetItem(os.path.basename(file_path)))
                self.file_table.setItem(row, 2, QTableWidgetItem(f"{size/1024/1024:.2f} MB"))
                self.file_table.setItem(row, 3, QTableWidgetItem(os.path.splitext(file_path)[1].upper()))
                self.file_table.setItem(row, 4, QTableWidgetItem(file_path))
            except OSError:
                continue
        
        self.stats_label.setText(f"Tổng số ảnh: {len(self.file_list)} | Tổng dung lượng: {total_size/1024/1024:.2f} MB")

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn file ảnh", "", "Image Files (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        if files:
            self.handle_dropped_files(files)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục")
        if folder:
            self.handle_dropped_files([folder])

    def select_all(self):
        for i in range(self.file_table.rowCount()):
            if item := self.file_table.item(i, 0):
                item.setCheckState(Qt.CheckState.Checked)

    def deselect_all(self):
        for i in range(self.file_table.rowCount()):
            if item := self.file_table.item(i, 0):
                item.setCheckState(Qt.CheckState.Unchecked)

    def reset_app(self):
        self.file_list.clear()
        self.update_file_table()
        self.stacked_widget.setCurrentIndex(0)

    def start_scan(self):
        if ScannerWorker is None:
            QMessageBox.critical(self, "Lỗi", "Không thể khởi tạo bộ quét (ScannerWorker). Vui lòng kiểm tra lại cấu hình và thử lại.")
            return

        files_to_scan = []
        for i in range(self.file_table.rowCount()):
            checkbox_item = self.file_table.item(i, 0)
            path_item = self.file_table.item(i, 4)
            if checkbox_item and path_item and checkbox_item.checkState() == Qt.CheckState.Checked:
                files_to_scan.append(path_item.text())

        if len(files_to_scan) < 2:
            QMessageBox.warning(self, "Không đủ file", "Vui lòng chọn ít nhất 2 file để quét.")
            return
        
        self.progress_dialog = QProgressDialog(self)
        self.progress_dialog.setWindowTitle("Đang xử lý")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        
        self.pause_resume_btn = QPushButton("Tạm dừng")
        self.pause_resume_btn.clicked.connect(self.toggle_pause_resume)
        
        layout = QVBoxLayout()
        self.progress_label = QLabel("Đang chuẩn bị quét...")
        self.progress_bar = QProgressBar()
        
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.pause_resume_btn)
        
        self.progress_dialog.setLayout(layout)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()

        self.scanner_worker = ScannerWorker(files_to_scan, self.similarity_slider.value())
        
        if self.scanner_worker:
            self.scanner_worker.progress_updated.connect(self.update_progress)
            self.scanner_worker.scan_completed.connect(self.scan_finished)
            self.scanner_worker.error_occurred.connect(self.scan_error)
            self.scanner_worker.finished.connect(self.progress_dialog.close)
            self.scanner_worker.start()
        else:
            self.progress_dialog.close()
            QMessageBox.critical(self, "Lỗi", "Không thể khởi tạo tiến trình quét.")

    def toggle_pause_resume(self):
        if self.scanner_worker:
            if self.pause_resume_btn.text() == "Tạm dừng":
                self.scanner_worker.pause()
                self.pause_resume_btn.setText("Tiếp tục")
            else:
                self.scanner_worker.resume()
                self.pause_resume_btn.setText("Tạm dừng")

    def update_progress(self, current, total, message):
        self.progress_label.setText(message)
        if total > 0:
            if self.progress_bar.minimum() != 0 or self.progress_bar.maximum() != total:
                self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setRange(0, 0)

    def scan_finished(self, results):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
        
        if not results:
            QMessageBox.information(self, "Hoàn tất", "Không tìm thấy ảnh nào trùng lặp hoặc tương đồng.")
            self.stacked_widget.setCurrentIndex(1)
        else:
            self.results_view.populate_results(results)
            self.stacked_widget.setCurrentIndex(2)

    def scan_error(self, error_message):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "Lỗi", f"Đã xảy ra lỗi trong quá trình quét:\n{error_message}")

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            'Xác nhận thoát',
            'Bạn có chắc chắn muốn thoát ứng dụng?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    try:
        import torch
    except ImportError:
        pass
    app = PixelPureApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    main_win = MainWindow()
    app.main_window = main_win
    main_win.show()
    sys.exit(app.exec())
