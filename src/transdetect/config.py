"""
config.py
-----------
Tham số cấu hình tập trung cho pipeline TransDetect-Vid.

Các giá trị mặc định ở đây khớp với mã giả và Listing trong Chương 3 của báo cáo
(kernel tiền xử lý, ngưỡng diện tích/tỉ lệ cạnh, ngưỡng conf/IoU của YOLO11).
"""

# --- Tiền xử lý (Mục 3.2) ---
MEDIAN_KERNEL_SIZE = 5

# --- Phát hiện truyền thống (Mục 3.3) ---
MIN_CONTOUR_AREA = 500
MAX_ASPECT_RATIO = 4.0
SOBEL_EDGE_THRESHOLD = 40

# --- YOLO11 (Mục 3.5) ---
DEFAULT_YOLO_MODEL = "yolo11n.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# --- Hiển thị ---
BOX_COLOR = (0, 255, 0)   # BGR - xanh lá
BOX_THICKNESS = 2
