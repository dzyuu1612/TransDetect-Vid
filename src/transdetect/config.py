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
MAX_CONTOUR_AREA = 150000
MAX_ASPECT_RATIO = 4.0
SOBEL_EDGE_THRESHOLD = 40

# --- YOLO11 (Mục 3.5) ---
DEFAULT_YOLO_MODEL = "yolo11n.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DET = 300

# --- Ánh xạ tên lớp về nhóm hiển thị (Mục 3.5.3) ---
# VEHICLE_NAMES nhận 6 tên để tương thích cả trọng số COCO lẫn model tùy biến,
# nhưng giao diện chỉ có 4 ô đếm; bảng dưới ánh xạ tên bổ sung về đúng nhóm.
CLASS_ALIAS = {
    "car": "Car",
    "motorcycle": "Motorcycle",
    "motorbike": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
    "container truck": "Truck",
}
DISPLAY_CLASSES = ("Car", "Motorcycle", "Bus", "Truck")

# --- Hiển thị ---
BOX_COLOR = (0, 255, 0)   # BGR - xanh lá
BOX_THICKNESS = 2
