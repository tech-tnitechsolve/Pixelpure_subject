# app_ui.py
#
# Phiên bản V32: Sửa lỗi ImportError.
# - [FIX] Di chuyển biến DARK_STYLESHEET ra khỏi khối `if __name__ == '__main__':`
#   để nó có thể được import đúng cách bởi các module khác.

import sys
import os
import time
from typing import Optional, Callable, cast, Any
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QStackedWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QScrollArea, QCheckBox,
    QProgressDialog, QProgressBar, QMessageBox, QFileDialog, QGraphicsOpacityEffect,
    QMenu
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QCloseEvent, QColor

# Thêm các import cần thiết
import torch
import open_clip
from open_clip import CLIP

from cache_manager import cache_manager
from improved_ui_components import ImprovedImageGroupWidget
from auto_processor import auto_processor
from result_dashboard import ResultDashboard
from core.scanner import ScannerWorker

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from send2trash import send2trash
except ImportError:
    send2trash = os.remove

# [FIX] Di chuyển DARK_STYLESHEET ra global scope
DARK_STYLESHEET = """
QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; }
QLabel { color: #ffffff; padding: 2px; }
QPushButton { background-color: #404040; border: 1px solid #606060; border-radius: 6px; padding: 8px 16px; color: #ffffff; font-weight: 500; }
QPushButton:hover { background-color: #505050; border-color: #707070; }
QPushButton:pressed { background-color: #303030; outline: none; border: 1px solid #555555; }
QPushButton:focus { outline: none; border: 1px solid #0078d4; }
QPushButton[cssClass="accent"] { background-color: #0078d4; border-color: #106ebe; }
QPushButton[cssClass="accent"]:hover { background-color: #106ebe; }
QPushButton[cssClass="accent"]:pressed { background-color: #005a9e; }
QTableWidget { background-color: #2d2d2d; alternate-background-color: #383838; gridline-color: #555555; border: 1px solid #606060; border-radius: 4px; }
QTableWidget::item { padding: 8px; border: none; }
QTableWidget::item:selected { background-color: #0078d4; }
QHeaderView::section { background-color: #404040; border: 1px solid #606060; padding: 8px; font-weight: 600; }
QProgressBar { border: 1px solid #606060; border-radius: 4px; text-align: center; background-color: #2d2d2d; }
QProgressBar::chunk { background-color: #0078d4; border-radius: 3px; }
QLabel[status="ok"] { color: #4caf50; font-weight: 600; }
QLabel[status="warn"] { color: #ff9800; font-weight: 600; }
QLabel[status="error"] { color: #f44336; font-weight: 600; }
"""

# --- Lớp tải Model trong nền ---
class ModelLoader(QThread):
    """Tải mô hình AI trong một luồng riêng để không làm treo UI."""
    model_loaded = Signal(object, object, object)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        from speed_config import SpeedConfig
        self.speed_config = SpeedConfig.get_mode('high_quality')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        try:
            model_name = self.speed_config['model_name']
            pretrained_tag = self.speed_config['model_pretrained']
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            model_cache_dir = os.path.join("core", "models")
            cache_manager.add_cache_dir(model_cache_dir)

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=model_name, pretrained=pretrained_tag, device=self.device, cache_dir=model_cache_dir
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            
            model.eval() # type: ignore
            if self.device == "cuda":
                model = model.half() # type: ignore
            
            self.model_loaded.emit(model, preprocess, tokenizer)
        except Exception as e:
            self.error_occurred.emit(f"Không thể tải AI Model: {e}")


class PixelPureApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setApplicationName("PixelPure")
        self.main_window: Optional['MainWindow'] = None

class DropZone(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.is_dragging = False
        self.click_animation = QPropertyAnimation(self, b"geometry")
        self.click_animation.setDuration(150)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        # Icon with animation effect
        self.icon_label = QLabel("📁")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 64px;")
        
        title_label = QLabel("PixelPure - Dọn dẹp thông minh")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        
        self.instruction_label = QLabel("Kéo thả file ảnh hoặc thư mục vào đây\nHoặc click để chọn file/thư mục")
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setStyleSheet("font-size: 14px; color: #cccccc; line-height: 1.5;")
        
        layout.addWidget(self.icon_label)
        layout.addWidget(title_label)
        layout.addWidget(self.instruction_label)
        
        # Thêm style cho khu vực có thể click (sửa lỗi transition)
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #555;
                border-radius: 10px;
                background-color: rgba(60, 60, 60, 0.3);
            }
            DropZone:hover {
                border-color: #0078d4;
                background-color: rgba(0, 120, 212, 0.1);
            }
        """)

    def mousePressEvent(self, event):
        """Handle click to open file selection directly"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Tạo hiệu ứng click
            self.animate_click()
            
            # Mở dialog chọn file trực tiếp
            self.select_files()

    def animate_click(self):
        """Tạo hiệu ứng khi click"""
        # Thay đổi icon và text tạm thời
        self.icon_label.setText("📂")
        self.instruction_label.setStyleSheet("font-size: 14px; color: #0078d4; line-height: 1.5;")
        
        # Reset lại sau 200ms
        QTimer.singleShot(200, self.reset_click_animation)

    def reset_click_animation(self):
        """Reset hiệu ứng click"""
        self.icon_label.setText("📁")
        self.instruction_label.setStyleSheet("font-size: 14px; color: #cccccc; line-height: 1.5;")

    def select_files(self):
        """Chọn file ảnh và folder"""
        # Mở dialog hỗ trợ cả file và folder
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "Chọn File Ảnh hoặc Thư Mục", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp);;All Files (*)"
        )
        
        # Nếu không chọn file, cho phép chọn folder
        if not files:
            folder = QFileDialog.getExistingDirectory(self, "Chọn Thư Mục")
            if folder:
                files = [folder]
        
        if files:
            self.handle_selected_paths(files)

    def select_folder(self):
        """Chọn thư mục"""
        folder = QFileDialog.getExistingDirectory(self, "Chọn Thư Mục")
        if folder:
            self.handle_selected_paths([folder])

    def handle_selected_paths(self, paths):
        """Xử lý đường dẫn được chọn"""
        widget = self
        while widget and not isinstance(widget, MainWindow):
            widget = widget.parent()
        if widget and hasattr(widget, 'handle_dropped_files'):
            widget.handle_dropped_files(paths)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.is_dragging = True
            self.update_drag_style(True)
            # Tạo hiệu ứng đường viền full màn hình
            self.create_fullscreen_border_effect()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.is_dragging = False
        self.update_drag_style(False)
        # Xóa hiệu ứng đường viền full màn hình
        self.remove_fullscreen_border_effect()
        super().dragLeaveEvent(event)

    def create_fullscreen_border_effect(self):
        """Tạo hiệu ứng đường viền full màn hình"""
        # Lấy main window
        main_window = self
        while main_window and not isinstance(main_window, MainWindow):
            main_window = main_window.parent()
        
        if main_window:
            # Thêm border effect cho main window
            main_window.setStyleSheet(main_window.styleSheet() + """
                MainWindow {
                    border: 3px solid #0078d4;
                    border-radius: 5px;
                }
            """)

    def remove_fullscreen_border_effect(self):
        """Xóa hiệu ứng đường viền full màn hình"""
        # Lấy main window
        main_window = self
        while main_window and not isinstance(main_window, MainWindow):
            main_window = main_window.parent()
        
        if main_window:
            # Reset style của main window
            current_style = main_window.styleSheet()
            # Xóa border effect
            new_style = current_style.replace("""
                MainWindow {
                    border: 3px solid #0078d4;
                    border-radius: 5px;
                }
            """, "")
            main_window.setStyleSheet(new_style)

    def update_drag_style(self, is_dragging):
        """Cập nhật style khi kéo thả"""
        if is_dragging:
            self.setStyleSheet("""
                DropZone {
                    border: 2px solid #0078d4;
                    border-radius: 10px;
                    background-color: rgba(0, 120, 212, 0.2);
                }
            """)
            self.icon_label.setText("📂")
            self.instruction_label.setText("Thả file/thư mục vào đây")
        else:
            self.setStyleSheet("""
                DropZone {
                    border: 2px dashed #555;
                    border-radius: 10px;
                    background-color: rgba(60, 60, 60, 0.3);
                }
                DropZone:hover {
                    border-color: #0078d4;
                    background-color: rgba(0, 120, 212, 0.1);
                }
            """)
            self.icon_label.setText("📁")
            self.instruction_label.setText("Kéo thả file ảnh hoặc thư mục vào đây\nHoặc click để chọn file/thư mục")

    def dropEvent(self, event):
        self.is_dragging = False
        self.update_drag_style(False)
        # Xóa hiệu ứng đường viền full màn hình
        self.remove_fullscreen_border_effect()
        
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.handle_selected_paths(paths)

class FileTableWidget(QTableWidget):
    def __init__(self):
        super().__init__()
        # Tắt selection để tránh đường viền đứt quãng
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        
        # Tối ưu hiệu suất
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        
        # Animation timer for click effect
        self.click_timer = QTimer()
        self.click_timer.setSingleShot(True)
        self.click_timer.timeout.connect(self.reset_click_effect)
        self.clicked_row = -1
        
    def reset_click_effect(self):
        """Reset click effect after animation"""
        if self.clicked_row >= 0:
            for col in range(self.columnCount()):
                item = self.item(self.clicked_row, col)
                if item:
                    item.setBackground(QColor())  # Reset to default
            self.clicked_row = -1
        
    def mousePressEvent(self, event):
        # Xử lý click để toggle checkbox và tạo hiệu ứng
        item = self.itemAt(event.pos())
        if item:
            row = item.row()
            checkbox_item = self.item(row, 0)
            if checkbox_item:
                # Reset previous click effect
                if self.clicked_row >= 0:
                    self.reset_click_effect()
                
                # Apply click effect
                self.clicked_row = row
                click_color = QColor(0, 120, 212, 60)  # Blue with transparency
                for col in range(self.columnCount()):
                    item_to_color = self.item(row, col)
                    if item_to_color:
                        item_to_color.setBackground(click_color)
                
                # Toggle checkbox state
                current_state = checkbox_item.checkState()
                new_state = Qt.CheckState.Unchecked if current_state == Qt.CheckState.Checked else Qt.CheckState.Checked
                checkbox_item.setCheckState(new_state)
                
                # Start timer to reset effect
                self.click_timer.start(300)  # 300ms effect
                
                # Trigger stats update in main window (với debounce)
                main_window = self
                while main_window and not isinstance(main_window, MainWindow):
                    main_window = main_window.parent()
                if main_window and hasattr(main_window, 'schedule_stats_update'):
                    main_window.schedule_stats_update()
        # Không gọi super() để tránh selection behavior

class ResultsView(QWidget):
    def __init__(self):
        super().__init__()
        self.group_widgets = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        header_layout = QHBoxLayout()
        title_label = QLabel("🔍 KẾT QUẢ QUÉT")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.scan_again_btn = QPushButton("🔄 Quét lại")
        self.scan_again_btn.setProperty("cssClass", "accent")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.scan_again_btn)
        layout.addLayout(header_layout)
        summary_layout = QHBoxLayout()
        summary_layout.setContentsMargins(8, 12, 8, 12)
        self.results_summary_label = QLabel("📊 Đang tải kết quả...")
        self.results_summary_label.setStyleSheet("QLabel { background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 rgba(56, 161, 105, 0.2), stop: 1 rgba(49, 130, 206, 0.2)); color: #e2e8f0; font-weight: 600; font-size: 13px; padding: 12px 20px; border-radius: 8px; border: 1px solid rgba(56, 161, 105, 0.3); font-family: 'Segoe UI', system-ui; }")
        self.execute_all_btn = QPushButton("⚡ TỰ ĐỘNG XỬ LÝ")
        self.execute_all_btn.setFixedHeight(45)
        self.execute_all_btn.setStyleSheet("QPushButton { background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #059669, stop: 0.5 #047857, stop: 1 #065f46); color: white; font-weight: bold; font-size: 12px; letter-spacing: 0.5px; padding: 12px 24px; border: none; border-radius: 8px; font-family: 'Segoe UI', system-ui; } QPushButton:hover { background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #047857, stop: 0.5 #065f46, stop: 1 #064e3b); } QPushButton:pressed { background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #064e3b, stop: 1 #047857); }")
        self.execute_all_btn.clicked.connect(self.execute_all_changes)
        summary_layout.addWidget(self.results_summary_label, 1)
        summary_layout.addSpacing(15)
        summary_layout.addWidget(self.execute_all_btn)
        layout.addLayout(summary_layout)
        info_layout = QHBoxLayout()
        threshold_info = QLabel("🧠 Subject Analysis V29+ NGHIÊM NGẶT (87% threshold)")
        threshold_info.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 11px;")
        info_layout.addWidget(threshold_info)
        info_layout.addStretch()
        layout.addLayout(info_layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: 1px solid rgba(74, 85, 104, 0.4); border-radius: 8px; background-color: rgba(26, 32, 44, 0.3); } QScrollBar:vertical { background-color: rgba(45, 55, 72, 0.5); width: 12px; border-radius: 6px; margin: 0px; } QScrollBar::handle:vertical { background-color: rgba(99, 179, 237, 0.6); border-radius: 6px; min-height: 20px; margin: 2px; } QScrollBar::handle:vertical:hover { background-color: rgba(99, 179, 237, 0.8); }")
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setSpacing(12)
        self.scroll_layout.setContentsMargins(12, 12, 12, 12)
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)

    def populate_results(self, results):
        # Lưu results để dùng cho tự động xử lý
        self.groups = results
        
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.group_widgets.clear()
        type_counts = {"duplicate": 0, "hybrid_subject": 0, "similar_subject": 0}
        for group in results:
            group_type = group.get('type', 'Unknown')
            type_counts[group_type] = type_counts.get(group_type, 0) + 1
            group_widget = ImprovedImageGroupWidget(group.get('files', []), group.get('score'), group_type, group.get('analysis_method', ''))
            group_widget.delete_clicked.connect(self.handle_delete)
            group_widget.move_clicked.connect(self.handle_move)
            self.group_widgets.append(group_widget)
            self.scroll_layout.addWidget(group_widget)
        summary_parts = []
        if (total_groups := len(results)) > 0: summary_parts.append(f"📊 <b>{total_groups}</b> nhóm")
        if (dup_count := type_counts["duplicate"]) > 0: summary_parts.append(f"🔄 <b style='color:#e53e3e'>{dup_count}</b> trùng lặp")
        if (hyb_count := type_counts["hybrid_subject"]) > 0: summary_parts.append(f"🧬 <b style='color:#9333ea'>{hyb_count}</b> hỗn hợp")
        if (sim_count := type_counts["similar_subject"]) > 0: summary_parts.append(f"🎯 <b style='color:#dd6b20'>{sim_count}</b> tương tự")
        self.results_summary_label.setText(" • ".join(summary_parts) if summary_parts else "📊 Không có kết quả.")
        self.scroll_layout.addStretch()

    def execute_all_changes(self):
        """Tự động xử lý tất cả nhóm theo logic:
        - Duplicates: Xóa các file trùng lặp, giữ lại 1 file gốc
        - Similars: Rename với pattern nhóm
        - Hybrids: Xóa duplicates + rename similars
        """
        if not hasattr(self, 'groups') or not self.groups:
            QMessageBox.warning(self, "Cảnh báo", "Không có kết quả scan để xử lý!")
            return
        
        # Hiển thị dialog xác nhận
        reply = QMessageBox.question(
            self, 
            "Xác nhận Tự động xử lý",
            f"Bạn có chắc muốn tự động xử lý {len(self.groups)} nhóm?\n\n"
            "🔴 Duplicates: Xóa file trùng lặp\n"
            "🟡 Similars: Đổi tên với pattern nhóm\n" 
            "🟣 Hybrids: Xóa duplicates + đổi tên similars\n\n"
            "⚠️ Thao tác này không thể hoàn tác!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        # Tạo progress dialog
        progress = QProgressDialog("Đang xử lý tự động...", "Hủy", 0, len(self.groups), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        processed_count = 0
        error_count = 0
        
        # Initialize sequential group counter for renaming
        self.group_counter = 0
        
        try:
            for i, group in enumerate(self.groups):
                if progress.wasCanceled():
                    break
                    
                progress.setValue(i)
                progress.setLabelText(f"Xử lý nhóm {i+1}/{len(self.groups)}: {group['type']}")
                QApplication.processEvents()
                
                try:
                    if group['type'] == 'duplicate':
                        self._process_duplicate_group(group)
                    elif group['type'] == 'similar_subject':
                        self._process_similar_group(group, i+1)  # Pass sequential group number
                    elif group['type'] == 'hybrid_subject':
                        self._process_hybrid_group(group, i+1)  # Pass sequential group number
                    
                    processed_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"Lỗi xử lý nhóm {i+1}: {e}")
            
            progress.setValue(len(self.groups))
            
            # Find parent MainWindow to call methods
            main_window = self.parent()
            while main_window and not isinstance(main_window, QMainWindow):
                main_window = main_window.parent()
            
            if main_window:
                # Clear results display through main window
                if hasattr(main_window, 'clear_results_display'):
                    cast(Any, main_window).clear_results_display()
                
                # Show completion dialog with option to rescan
                result = QMessageBox.question(
                    self,
                    "🎉 Xử lý hoàn thành!",
                    f"✅ Đã xử lý: {processed_count}/{len(self.groups)} nhóm\n"
                    f"❌ Lỗi: {error_count} nhóm\n\n"
                    "🔍 Bạn có muốn quét lại để kiểm tra các file còn lại không?\n"
                    "(Đề xuất: Có thể còn file tương tự chưa được phát hiện)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if result == QMessageBox.StandardButton.Yes:
                    # Auto-trigger rescan with remaining files
                    if hasattr(main_window, 'auto_rescan_after_processing'):
                        cast(Any, main_window).auto_rescan_after_processing()
            else:
                # Fallback if can't find main window
                QMessageBox.information(
                    self,
                    "🎉 Xử lý hoàn thành!",
                    f"✅ Đã xử lý: {processed_count}/{len(self.groups)} nhóm\n"
                    f"❌ Lỗi: {error_count} nhóm\n\n"
                    "Vui lòng kiểm tra kết quả trong thư mục!"
                )
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi trong quá trình xử lý: {e}")
        
        finally:
            progress.close()

    def _process_duplicate_group(self, group):
        """Xử lý nhóm duplicate: xóa files nhỏ hơn, giữ lại file có size lớn nhất"""
        import send2trash
        import os
        
        files = group['files']
        print(f"DEBUG: Processing DUPLICATE group with {len(files)} files")
        
        # Extract file paths và size từ dict structure
        file_data = []
        for f in files:
            if isinstance(f, dict) and 'path' in f:
                path = os.path.normpath(f['path'])
                size = f.get('size', 0)
                file_data.append({'path': path, 'size': size})
                print(f"DEBUG: File: {os.path.basename(path)} - Size: {size} bytes")
        
        if len(file_data) <= 1:
            return
            
        # Sắp xếp theo size giảm dần - file lớn nhất ở đầu
        file_data.sort(key=lambda x: x['size'], reverse=True)
        
        # Giữ lại file có size lớn nhất (đầu tiên sau sort)
        keep_file = file_data[0]
        files_to_delete = file_data[1:]
        
        print(f"DEBUG: KEEPING largest file: {os.path.basename(keep_file['path'])} ({keep_file['size']} bytes)")
        
        for file_info in files_to_delete:
            file_path = file_info['path']  # Move outside try block
            try:
                if os.path.exists(file_path):
                    print(f"DEBUG: DELETING smaller file: {os.path.basename(file_path)} ({file_info['size']} bytes)")
                    send2trash.send2trash(file_path)
                else:
                    print(f"DEBUG: File not found: {file_path}")
            except Exception as e:
                print(f"DEBUG: Error deleting {file_path}: {e}")
                # Fallback to try different path format
                try:
                    alt_path = file_path.replace('/', '\\')
                    if os.path.exists(alt_path):
                        send2trash.send2trash(alt_path)
                except:
                    pass

    def _process_similar_group(self, group, group_num=None):
        """Xử lý nhóm similar: đổi tên theo pattern nhóm X(1), X(2), X(3)..."""
        files = group['files']
        
        print(f"DEBUG: Processing SIMILAR group with {len(files)} files")
        
        # Extract file paths từ dict structure
        file_paths = []
        for f in files:
            if isinstance(f, dict) and 'path' in f:
                path = os.path.normpath(f['path'])
                file_paths.append(path)
                
        if len(file_paths) <= 1:
            return
            
        # Use provided group number or generate one
        if group_num is None:
            group_num = hash(str(sorted(file_paths))) % 9999 + 1  # 1-9999
        
        print(f"DEBUG: Renaming SIMILAR group {group_num} with {len(file_paths)} files")
        
        for i, file_path in enumerate(file_paths):
            try:
                if not os.path.exists(file_path):
                    continue
                    
                dir_path = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)
                name, ext = os.path.splitext(file_name)
                
                # New naming pattern: GroupNumber(FileNumber)
                new_name = f"{group_num}({i+1}){ext}"
                new_path = os.path.join(dir_path, new_name)
                
                # Tránh ghi đè file và kiểm tra độ dài filename
                counter = 1
                while os.path.exists(new_path) or len(new_path) > 250:
                    if len(new_path) > 250:
                        # Shorten if too long
                        new_name = f"{group_num}({i+1})_{counter}{ext}"
                        new_path = os.path.join(dir_path, new_name)
                    else:
                        new_name = f"{group_num}({i+1})_{counter}{ext}"
                        new_path = os.path.join(dir_path, new_name)
                    counter += 1
                    if counter > 100:
                        break
                
                print(f"DEBUG: RENAMING SIMILAR: {os.path.basename(file_path)} -> {new_name}")
                os.rename(file_path, new_path)
            except Exception as e:
                print(f"DEBUG: Error renaming {file_path}: {e}")

    def _process_hybrid_group(self, group, group_num=None):
        """Xử lý nhóm hybrid: xóa duplicates (giữ file lớn nhất), rename similars theo pattern X(1), X(2)"""
        import send2trash
        
        files = group['files']
        
        print(f"DEBUG: Processing HYBRID group with {len(files)} files")
        
        # Extract file paths và analyze roles
        duplicates = []
        similars = []
        
        for f in files:
            if isinstance(f, dict) and 'path' in f:
                path = os.path.normpath(f['path'])
                role = f.get('role_in_group', 'unknown')
                size = f.get('size', 0)
                
                if role == 'duplicate':
                    duplicates.append({'path': path, 'size': size})
                else:
                    similars.append({'path': path, 'size': size})
        
        print(f"DEBUG: Found {len(duplicates)} duplicates, {len(similars)} similars")
        
        # Step 1: Xử lý duplicates - giữ file lớn nhất, xóa các file nhỏ hơn
        if len(duplicates) > 1:
            duplicates.sort(key=lambda x: x['size'], reverse=True)
            keep_duplicate = duplicates[0]
            delete_duplicates = duplicates[1:]
            
            print(f"DEBUG: KEEPING largest duplicate: {os.path.basename(keep_duplicate['path'])} ({keep_duplicate['size']} bytes)")
            
            for dup_info in delete_duplicates:
                file_path = dup_info['path']
                try:
                    if os.path.exists(file_path):
                        print(f"DEBUG: DELETING smaller duplicate: {os.path.basename(file_path)} ({dup_info['size']} bytes)")
                        send2trash.send2trash(file_path)
                except Exception as e:
                    print(f"DEBUG: Error deleting duplicate {file_path}: {e}")
            
            # Thêm file duplicate được giữ lại vào danh sách similars để rename
            similars.append(keep_duplicate)
        elif len(duplicates) == 1:
            # Chỉ có 1 duplicate, thêm vào similars để rename
            similars.append(duplicates[0])
        
        # Step 2: Rename tất cả similars (bao gồm duplicate được giữ lại) theo pattern X(1), X(2)
        if len(similars) > 1:
            # Use provided group number or generate one
            if group_num is None:
                all_paths = [s['path'] for s in similars]
                group_num = hash(str(sorted(all_paths))) % 9999 + 1
            
            print(f"DEBUG: Renaming HYBRID group {group_num} with {len(similars)} files")
            
            for i, sim_info in enumerate(similars):
                file_path = sim_info['path']
                try:
                    if not os.path.exists(file_path):
                        continue
                        
                    dir_path = os.path.dirname(file_path)
                    file_name = os.path.basename(file_path)
                    name, ext = os.path.splitext(file_name)
                    
                    # New naming pattern: GroupNumber(FileNumber)
                    new_name = f"{group_num}({i+1}){ext}"
                    new_path = os.path.join(dir_path, new_name)
                    
                    # Tránh ghi đè file
                    counter = 1
                    while os.path.exists(new_path) or len(new_path) > 250:
                        if len(new_path) > 250:
                            new_name = f"{group_num}({i+1})_{counter}{ext}"
                            new_path = os.path.join(dir_path, new_name)
                        else:
                            new_name = f"{group_num}({i+1})_{counter}{ext}"
                            new_path = os.path.join(dir_path, new_name)
                        counter += 1
                        if counter > 100:
                            break
                    
                    print(f"DEBUG: RENAMING HYBRID: {os.path.basename(file_path)} -> {new_name}")
                    os.rename(file_path, new_path)
                except Exception as e:
                    print(f"DEBUG: Error renaming hybrid {file_path}: {e}")

    def handle_delete(self, group_widget): pass
    def handle_move(self, group_widget): pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixelPure - Dọn dẹp thông minh, Tối ưu không gian")
        self.setGeometry(100, 100, 1200, 800)
        self.file_list = set()
        self.scanner_worker = None
        self.progress_dialog = None 
        
        self.loaded_model: Optional[CLIP] = None
        self.loaded_preprocess: Optional[Callable] = None
        self.loaded_tokenizer: Optional[Callable] = None

        # Timer for debounced stats update
        self.stats_update_timer = QTimer()
        self.stats_update_timer.setSingleShot(True)
        self.stats_update_timer.timeout.connect(self.update_stats_label)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.create_drop_zone_view()
        self.create_file_list_view()
        self.results_view = ResultsView()
        self.stacked_widget.addWidget(self.results_view)
        self.results_view.scan_again_btn.clicked.connect(self.reset_app)
        self.stacked_widget.setCurrentIndex(0)
        
        self.check_gpu_on_startup()
        self.start_model_loading()

    def schedule_stats_update(self):
        """Schedule stats update with debounce to improve performance"""
        self.stats_update_timer.start(50)  # 50ms debounce

    def start_model_loading(self):
        self.model_loader = ModelLoader()
        self.model_loader.model_loaded.connect(self.on_model_loaded)
        self.model_loader.error_occurred.connect(self.on_model_load_error)
        self.model_loader.start()

    def on_model_loaded(self, model, preprocess, tokenizer):
        self.loaded_model = model
        self.loaded_preprocess = preprocess
        self.loaded_tokenizer = tokenizer
        self.start_scan_btn.setEnabled(True)
        self.start_scan_btn.setToolTip("Sẵn sàng để quét!")
        self.update_device_status()
        
    def on_model_load_error(self, error_message):
        self.device_status_label.setText(f"❌ LỖI MODEL: {error_message}")
        self.device_status_label.setProperty("status", "error")
        QMessageBox.critical(self, "Lỗi tải Model", error_message)

    def check_gpu_on_startup(self):
        try:
            if not torch.cuda.is_available():
                QMessageBox.warning(self, "⚠️ Cảnh báo GPU", "Không tìm thấy GPU tương thích CUDA. Ứng dụng sẽ chạy trên CPU và có thể chậm hơn.", QMessageBox.StandardButton.Ok)
        except ImportError:
             QMessageBox.critical(self, "Lỗi nghiêm trọng", "Không tìm thấy PyTorch. Vui lòng chạy `install_requirements.py`.", QMessageBox.StandardButton.Ok)

    def create_drop_zone_view(self):
        drop_zone = DropZone()
        self.stacked_widget.addWidget(drop_zone)

    def create_file_list_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
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
        self.file_table = FileTableWidget()
        self.file_table.setColumnCount(5)
        self.file_table.setHorizontalHeaderLabels(["", "Tên File", "Kích Thước", "Định Dạng", "Đường Dẫn"])
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.file_table.setColumnWidth(0, 30)
        
        bottom_bar = QHBoxLayout()
        self.stats_label = QLabel("Tổng số ảnh: 0")
        self.device_status_label = QLabel("🤖 Đang tải AI Model...")
        self.device_status_label.setProperty("status", "warn")
        self.time_estimate_label = QLabel("")
        self.time_estimate_label.setStyleSheet("color: #4caf50; font-style: italic;")
        self.reset_btn = QPushButton("Reset")
        self.start_scan_btn = QPushButton("Bắt Đầu Quét")
        self.start_scan_btn.setProperty("cssClass", "accent")
        self.start_scan_btn.setEnabled(False)
        self.start_scan_btn.setToolTip("Vui lòng đợi AI Model tải xong...")
        
        bottom_bar.addWidget(self.stats_label)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.device_status_label)
        bottom_bar.addSpacing(20)
        bottom_bar.addWidget(self.time_estimate_label)
        bottom_bar.addSpacing(20)
        bottom_bar.addWidget(self.reset_btn)
        bottom_bar.addWidget(self.start_scan_btn)
        
        layout.addLayout(top_bar)
        layout.addWidget(self.file_table)
        layout.addLayout(bottom_bar)
        
        self.stacked_widget.addWidget(widget)
        self.add_file_btn.clicked.connect(self.add_files)
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.select_all_btn.clicked.connect(self.select_all)
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        self.reset_btn.clicked.connect(self.reset_app)
        self.start_scan_btn.clicked.connect(self.start_scan)
        self.file_table.itemChanged.connect(self.on_item_changed)
        self.update_time_estimate()

    def on_item_changed(self, item):
        """Handle when any item in the table changes"""
        self.schedule_stats_update()

    def update_device_status(self):
        try:
            if torch.cuda.is_available():
                self.device_status_label.setText(f"✅ AI Sẵn Sàng (GPU: {torch.cuda.get_device_name(0)})")
                self.device_status_label.setProperty("status", "ok")
            else:
                self.device_status_label.setText("✅ AI Sẵn Sàng (CPU)")
                self.device_status_label.setProperty("status", "ok")
        except ImportError:
            self.device_status_label.setText("❌ Lỗi PyTorch")
            self.device_status_label.setProperty("status", "error")
        
        self.device_status_label.style().unpolish(self.device_status_label)
        self.device_status_label.style().polish(self.device_status_label)
    
    def update_time_estimate(self, item=None):
        try:
            from speed_config import SpeedConfig
            file_count = sum(1 for i in range(self.file_table.rowCount()) if (cb_item := self.file_table.item(i, 0)) and cb_item.checkState() == Qt.CheckState.Checked)
            if file_count > 0:
                self.time_estimate_label.setText(f"⏱️ Ước tính: {SpeedConfig.estimate_time(file_count)}")
            else:
                self.time_estimate_label.setText("")
        except ImportError:
            self.time_estimate_label.setText("")

    def handle_dropped_files(self, paths):
        valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        for path in paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for name in files:
                        if os.path.splitext(name)[1].lower() in valid_ext:
                            self.file_list.add(os.path.join(root, name))
            elif os.path.isfile(path) and os.path.splitext(path)[1].lower() in valid_ext:
                self.file_list.add(path)
        self.update_file_table()
        if self.file_list: self.stacked_widget.setCurrentIndex(1)

    def format_size(self, size_bytes):
        """Format file size to appropriate unit (B, KB, MB, GB, TB)"""
        if size_bytes == 0:
            return "0 B"
        
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        size = float(size_bytes)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        if unit_index == 0:  # Bytes
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.2f} {units[unit_index]}"

    def update_stats_label(self):
        """Update stats label with selected/total count and size with color indicators"""
        total_files = self.file_table.rowCount()
        selected_files = 0
        selected_size = 0
        total_size = 0
        
        for i in range(total_files):
            checkbox_item = self.file_table.item(i, 0)
            path_item = self.file_table.item(i, 4)
            
            if path_item:
                try:
                    file_size = os.path.getsize(path_item.text())
                    total_size += file_size
                    
                    if checkbox_item and checkbox_item.checkState() == Qt.CheckState.Checked:
                        selected_files += 1
                        selected_size += file_size
                except OSError:
                    continue
        
        # Determine color based on selection
        if selected_files == 0:
            color = "#f44336"  # Red - no files selected
            status = "Chưa chọn file nào"
        elif selected_files < 2:
            color = "#ff9800"  # Orange - insufficient files for scanning
            status = "Cần ít nhất 2 file để quét"
        elif selected_files == total_files:
            color = "#4caf50"  # Green - all files selected
            status = "Đã chọn tất cả"
        else:
            color = "#2196f3"  # Blue - partial selection
            status = "Đã chọn một phần"
        
        # Update stats label with color
        self.stats_label.setText(
            f"<span style='color: {color}; font-weight: bold;'>{status}</span> | "
            f"Đã chọn: <span style='color: {color}; font-weight: bold;'>{selected_files}/{total_files}</span> | "
            f"Dung lượng: <span style='color: {color};'>{self.format_size(selected_size)}/{self.format_size(total_size)}</span>"
        )
        
        # Also update time estimate
        self.update_time_estimate()

    def update_file_table(self):
        """Update file table with performance optimization"""
        self.file_table.blockSignals(True)
        self.file_table.setSortingEnabled(False)  # Tắt sorting khi update
        self.file_table.setRowCount(0)
        
        # Batch insert rows for better performance
        file_list = sorted(list(self.file_list))
        self.file_table.setRowCount(len(file_list))
        
        for row, file_path in enumerate(file_list):
            try:
                # Checkbox
                chk = QTableWidgetItem()
                chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                chk.setCheckState(Qt.CheckState.Checked)
                self.file_table.setItem(row, 0, chk)
                
                # File info
                size = os.path.getsize(file_path)
                self.file_table.setItem(row, 1, QTableWidgetItem(os.path.basename(file_path)))
                self.file_table.setItem(row, 2, QTableWidgetItem(self.format_size(size)))
                self.file_table.setItem(row, 3, QTableWidgetItem(os.path.splitext(file_path)[1].upper()))
                self.file_table.setItem(row, 4, QTableWidgetItem(file_path))
            except OSError: 
                continue
        
        self.file_table.setSortingEnabled(True)  # Bật lại sorting
        self.file_table.blockSignals(False)
        self.schedule_stats_update()

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn file ảnh", "", "Image Files (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        if files: self.handle_dropped_files(files)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục")
        if folder: self.handle_dropped_files([folder])

    def select_all(self):
        self.file_table.blockSignals(True)  # Tối ưu hiệu suất
        for i in range(self.file_table.rowCount()):
            if (item := self.file_table.item(i, 0)): 
                item.setCheckState(Qt.CheckState.Checked)
        self.file_table.blockSignals(False)
        self.schedule_stats_update()

    def deselect_all(self):
        self.file_table.blockSignals(True)  # Tối ưu hiệu suất
        for i in range(self.file_table.rowCount()):
            if (item := self.file_table.item(i, 0)): 
                item.setCheckState(Qt.CheckState.Unchecked)
        self.file_table.blockSignals(False)
        self.schedule_stats_update()

    def reset_app(self):
        self.file_list.clear()
        self.update_file_table()
        self.stacked_widget.setCurrentIndex(0)

    def toggle_pause_resume(self):
        if self.scanner_worker and self.progress_dialog:
            pause_resume_btn = self.progress_dialog.findChild(QPushButton)
            if pause_resume_btn:
                if pause_resume_btn.text() == "Tạm dừng":
                    self.scanner_worker.pause()
                    pause_resume_btn.setText("Tiếp tục")
                else:
                    self.scanner_worker.resume()
                    pause_resume_btn.setText("Tạm dừng")

    def start_scan(self):
        if not all([self.loaded_model, self.loaded_preprocess, self.loaded_tokenizer]):
            QMessageBox.information(self, "Vui lòng đợi", "Mô hình AI vẫn đang được tải. Vui lòng thử lại sau giây lát.")
            return

        files_to_scan = []
        for i in range(self.file_table.rowCount()):
            cb_item = self.file_table.item(i, 0)
            path_item = self.file_table.item(i, 4)
            if cb_item and path_item and cb_item.checkState() == Qt.CheckState.Checked:
                files_to_scan.append(path_item.text())

        if len(files_to_scan) < 2:
            QMessageBox.warning(self, "Không đủ file", "Vui lòng chọn ít nhất 2 file để quét.")
            return
        
        self.progress_dialog = QProgressDialog(self)
        self.progress_dialog.setWindowTitle("🔍 Đang quét và phân tích")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumWidth(450)
        self.progress_dialog.setMinimumHeight(200)
        
        # Create custom layout for progress dialog
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Main status label
        progress_label = QLabel("🚀 Chuẩn bị khởi động quét...")
        progress_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196f3;")
        
        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #4caf50, stop: 1 #2196f3);
                border-radius: 6px;
            }
        """)
        
        # Detailed info labels
        info_layout = QHBoxLayout()
        self.scan_info_label = QLabel(f"📊 Tổng file: {len(files_to_scan)}")
        self.speed_info_label = QLabel("⚡ Tốc độ: Đang tính...")
        self.scan_info_label.setStyleSheet("color: #666; font-size: 11px;")
        self.speed_info_label.setStyleSheet("color: #666; font-size: 11px;")
        info_layout.addWidget(self.scan_info_label)
        info_layout.addStretch()
        info_layout.addWidget(self.speed_info_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        pause_resume_btn = QPushButton("⏸️ Tạm dừng")
        pause_resume_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        button_layout.addStretch()
        button_layout.addWidget(pause_resume_btn)
        
        layout.addWidget(progress_label)
        layout.addWidget(progress_bar)
        layout.addLayout(info_layout)
        layout.addLayout(button_layout)
        
        self.progress_dialog.setLayout(layout)
        self.progress_dialog.setCancelButton(None)
        
        model = cast(CLIP, self.loaded_model)
        preprocess = cast(Callable, self.loaded_preprocess)
        tokenizer = cast(Callable, self.loaded_tokenizer)
        
        self.scanner_worker = ScannerWorker(files_to_scan, 0.87, model, preprocess, tokenizer)
        
        pause_resume_btn.clicked.connect(self.toggle_pause_resume)
        
        # Enhanced progress callback
        def update_progress(current, total, message):
            progress_label.setText(f"🔍 {message}")
            progress_bar.setRange(0, total)
            progress_bar.setValue(current)
            
            # Update detailed info
            if current > 0:
                percentage = (current / total) * 100
                remaining = total - current
                self.scan_info_label.setText(f"📊 Tiến độ: {current}/{total} ({percentage:.1f}%)")
                
                if (self.scanner_worker and 
                    hasattr(self.scanner_worker, 'start_time') and 
                    self.scanner_worker.start_time):
                    elapsed = time.time() - self.scanner_worker.start_time
                    if elapsed > 1 and current > 1:  # Avoid division by zero
                        speed = current / elapsed
                        eta = remaining / speed if speed > 0 else 0
                        self.speed_info_label.setText(f"⚡ Còn lại: {eta:.0f}s (~{speed:.1f} file/s)")
        
        self.scanner_worker.progress_updated.connect(update_progress)
        self.scanner_worker.scan_completed.connect(self.scan_finished)
        self.scanner_worker.error_occurred.connect(self.scan_error)
        self.scanner_worker.finished.connect(self.progress_dialog.close)
        
        self.progress_dialog.show()
        self.scanner_worker.start()

    def scan_finished(self, results):
        # Cleanup thread properly
        if self.scanner_worker:
            self.scanner_worker.wait(1000)  # Wait for thread to finish
            self.scanner_worker.deleteLater()  # Schedule for deletion
            self.scanner_worker = None
        
        if not results:
            QMessageBox.information(self, "Hoàn tất", "Không tìm thấy ảnh trùng lặp hoặc tương đồng.")
            self.stacked_widget.setCurrentIndex(1)
        else:
            self.results_view.populate_results(results)
            self.stacked_widget.setCurrentIndex(2)

    def scan_error(self, error_message):
        # Cleanup thread on error
        if self.scanner_worker:
            self.scanner_worker.wait(1000)
            self.scanner_worker.deleteLater()
            self.scanner_worker = None
            
        QMessageBox.critical(self, "Lỗi", f"Đã xảy ra lỗi trong quá trình quét:\n{error_message}")

    def closeEvent(self, event: QCloseEvent):
        reply = QMessageBox.question(self, 'Xác nhận thoát', 'Bạn có chắc chắn muốn thoát?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.scanner_worker and self.scanner_worker.isRunning():
                self.scanner_worker.stop()
                self.scanner_worker.wait(3000)
            cache_manager.full_cleanup()
            event.accept()
        else:
            event.ignore()

    def clear_results_display(self):
        """Clear current results display and reset to empty state"""
        # Clear all group widgets from the results view
        if hasattr(self, 'results_view') and self.results_view:
            # Clear scroll layout
            while self.results_view.scroll_layout.count() > 0:
                child = self.results_view.scroll_layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)
            
            # Reset results summary
            self.results_view.results_summary_label.setText("📊 Kết quả đã được xử lý - sẵn sàng quét lại")
            
            # Clear groups data
            if hasattr(self.results_view, 'groups'):
                self.results_view.groups = []
        
        print("DEBUG: Results display cleared")

    def auto_rescan_after_processing(self):
        """Automatically rescan remaining files after processing to find missed similarities"""
        try:
            # Get remaining files from file list (files that still exist)
            remaining_files = []
            for i in range(self.file_table.rowCount()):
                path_item = self.file_table.item(i, 4)
                if path_item:
                    file_path = path_item.text()
                    if os.path.exists(file_path):
                        remaining_files.append(file_path)
            
            if len(remaining_files) < 2:
                QMessageBox.information(
                    self, 
                    "Quét lại hoàn tất", 
                    "📋 Không còn đủ file để quét lại.\n✅ Quá trình xử lý đã hoàn thành!"
                )
                return
            
            # Update file table with remaining files
            self.file_list = set(remaining_files)
            self.update_file_table()
            
            # Show info about rescan
            reply = QMessageBox.question(
                self,
                "🔍 Quét lại để kiểm tra",
                f"📊 Phát hiện {len(remaining_files)} file còn lại\n\n"
                "🎯 Sẽ thực hiện quét lại với logic cải tiến để:\n"
                "• Phát hiện các file tương tự bị bỏ sót\n"
                "• Kiểm tra các nhóm có thể bị gộp nhầm\n"
                "• Áp dụng threshold nghiêm ngặt hơn\n\n"
                "Bạn có muốn tiếp tục không?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Select all remaining files and start enhanced scan
                self.select_all()
                self.start_enhanced_rescan()
            
        except Exception as e:
            print(f"DEBUG: Error in auto_rescan_after_processing: {e}")
            QMessageBox.warning(self, "Lỗi", f"Có lỗi khi quét lại: {e}")

    def start_enhanced_rescan(self):
        """Start enhanced rescan with improved similarity detection"""
        if not all([self.loaded_model, self.loaded_preprocess, self.loaded_tokenizer]):
            QMessageBox.information(self, "Vui lòng đợi", "Mô hình AI vẫn đang được tải. Vui lòng thử lại sau giây lát.")
            return

        files_to_scan = []
        for i in range(self.file_table.rowCount()):
            cb_item = self.file_table.item(i, 0)
            path_item = self.file_table.item(i, 4)
            if cb_item and path_item and cb_item.checkState() == Qt.CheckState.Checked:
                files_to_scan.append(path_item.text())

        if len(files_to_scan) < 2:
            QMessageBox.warning(self, "Không đủ file", "Cần ít nhất 2 file để quét lại.")
            return
        
        # Use existing start_scan method - it will automatically use enhanced logic in scanner
        self.start_scan()

    def on_enhanced_rescan_finished(self, results):
        """Handle completion of enhanced rescan"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if self.scanner_worker:
            self.scanner_worker.wait()
            self.scanner_worker.deleteLater()
            self.scanner_worker = None
        
        if results:
            # Display results in the results view
            if hasattr(self, 'results_view') and self.results_view:
                self.results_view.populate_results(results)
                self.stacked_widget.setCurrentIndex(2)
            
            QMessageBox.information(
                self,
                "🔍 Quét lại hoàn tất",
                f"✅ Phát hiện thêm {len(results)} nhóm có thể bị bỏ sót!\n\n"
                "🎯 Vui lòng kiểm tra kết quả và xử lý nếu cần."
            )
        else:
            QMessageBox.information(
                self,
                "✅ Quét lại hoàn tất", 
                "🎉 Không phát hiện thêm nhóm nào!\n"
                "✨ Quá trình xử lý đã hoàn thành hoàn hảo."
            )

if __name__ == '__main__':
    app = PixelPureApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    main_win = MainWindow()
    app.main_window = main_win
    main_win.show()
    sys.exit(app.exec())
