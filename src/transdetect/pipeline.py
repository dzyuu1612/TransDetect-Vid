"""
pipeline.py
-------------
Gom hai nhánh xử lý video theo đúng sơ đồ Mục 3.1 của báo cáo.

- Nhánh truyền thống: đọc frame -> tiền xử lý -> threshold/Sobel -> contour/box
  -> Lucas-Kanade tracking -> ghi video.
- Nhánh YOLO11: đọc frame -> YOLO11 inference -> lọc lớp phương tiện -> vẽ box
  -> ghi video.
"""

import os

import cv2

from . import config
from . import preprocessing
from . import classical_detector
from . import visualization
from .optical_flow import LucasKanadeTracker
from .yolo_detector import Yolo11VehicleDetector


def _open_capture(input_path):
    """Mở video đầu vào và bảo đảm đọc được TRƯỚC khi vào vòng lặp frame.

    Không có bước kiểm tra này, một đường dẫn sai sẽ khiến cap.read() trả về
    False ngay ở frame đầu tiên: vòng lặp thoát lập tức và chương trình in ra
    "Đã lưu video kết quả" dù file kết quả rỗng.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video đầu vào: {input_path}")
    return cap


def _open_writer(cap, output_path):
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Kích thước phải hợp lệ, vì VideoWriter chỉ ghi được frame đúng bằng
    # (width, height) đã khai báo; sai kích thước thì write() không làm gì cả.
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(
            f"Video đầu vào báo kích thước không hợp lệ: {width}x{height}"
        )

    # Thư mục đích phải tồn tại sẵn, nếu không VideoWriter thất bại im lặng.
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Không khởi tạo được VideoWriter cho: {output_path}")
    return writer


def run_classical(input_path, output_path):
    """Chạy nhánh truyền thống trên toàn bộ video."""
    cap = _open_capture(input_path)
    writer = _open_writer(cap, output_path)
    tracker = LucasKanadeTracker()
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pre = preprocessing.preprocess_frame(frame, config.MEDIAN_KERNEL_SIZE)
        boxes, _, _ = classical_detector.detect_vehicle_candidates(
            pre,
            min_area=config.MIN_CONTOUR_AREA,
            max_area=config.MAX_CONTOUR_AREA,
            max_aspect_ratio=config.MAX_ASPECT_RATIO,
            edge_threshold=config.SOBEL_EDGE_THRESHOLD,
        )
        output = visualization.draw_classical_boxes(frame, boxes)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            points, motion = tracker.track_features(prev_gray, curr_gray)
            if len(points) > 0:
                output = visualization.draw_motion_vectors(output, points, motion)
        prev_gray = curr_gray

        writer.write(output)

    cap.release()
    writer.release()
    print(f"Đã lưu video kết quả tại: {output_path}")


def run_yolo(input_path, output_path, model_path=None, conf=None, iou=None,
             max_det=None):
    """Chạy nhánh YOLO11 trên toàn bộ video."""
    detector = Yolo11VehicleDetector(model_path or config.DEFAULT_YOLO_MODEL)
    conf = config.CONF_THRESHOLD if conf is None else conf
    iou = config.IOU_THRESHOLD if iou is None else iou
    max_det = config.MAX_DET if max_det is None else max_det

    cap = _open_capture(input_path)
    writer = _open_writer(cap, output_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_frame(frame, conf, iou, max_det)
        output = visualization.draw_yolo_detections(frame, detections)
        writer.write(output)

    cap.release()
    writer.release()
    print(f"Đã lưu video kết quả tại: {output_path}")
