# main.py
#
# Điểm khởi chạy chính của ứng dụng PixelPure.
# Chạy file này để bắt đầu chương trình:
# python main.py

import sys
# Sửa lỗi: Import lớp PixelPureApplication và MainWindow từ app_ui
from app_ui import PixelPureApplication, MainWindow, DARK_STYLESHEET

if __name__ == "__main__":
    # Sửa lỗi: Khởi tạo ứng dụng bằng lớp PixelPureApplication tùy chỉnh
    # thay vì QApplication tiêu chuẩn.
    app = PixelPureApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    
    # Tạo và hiển thị cửa sổ chính
    main_win = MainWindow()
    
    # Gán cửa sổ chính vào thuộc tính của ứng dụng.
    # Pylance sẽ hiểu thuộc tính này vì nó được định nghĩa trong PixelPureApplication.
    app.main_window = main_win
    main_win.show()
    
    # Bắt đầu vòng lặp sự kiện của ứng dụng và thoát khi cửa sổ đóng
    sys.exit(app.exec())