"""
pipeline.py
-------------
Gom hai nhánh xử lý video theo đúng sơ đồ Mục 3.1 của báo cáo.

- Nhánh truyền thống: đọc frame -> tiền xử lý -> threshold/Sobel -> contour/box
  -> Lucas-Kanade tracking -> ghi video.
- Nhánh YOLO11: đọc frame -> YOLO11 inference -> lọc lớp phương tiện -> vẽ box
  -> ghi video.
"""

import cv2
import numpy as np

from . import config
from . import preprocessing
from . import classical_detector
from . import visualization
from .optical_flow import LucasKanadeTracker
from .yolo_detector import Yolo11VehicleDetector


def _open_writer(cap, output_path):
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def run_classical(input_path, output_path):
    """Chạy nhánh truyền thống trên toàn bộ video."""
    cap = cv2.VideoCapture(input_path)
    writer = _open_writer(cap, output_path)
    tracker = LucasKanadeTracker()
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pre = preprocessing.preprocess_frame(frame, config.MEDIAN_KERNEL_SIZE)
        boxes, _, _, _ = classical_detector.detect_vehicle_candidates(
            pre, config.MIN_CONTOUR_AREA, config.MAX_ASPECT_RATIO
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


def run_yolo(input_path, output_path, model_path=None, conf=None, iou=None):
    """Chạy nhánh YOLO11 trên toàn bộ video."""
    detector = Yolo11VehicleDetector(model_path or config.DEFAULT_YOLO_MODEL)
    conf = config.CONF_THRESHOLD if conf is None else conf
    iou = config.IOU_THRESHOLD if iou is None else iou

    cap = cv2.VideoCapture(input_path)
    writer = _open_writer(cap, output_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_frame(frame, conf, iou)
        output = visualization.draw_yolo_detections(frame, detections)
        writer.write(output)

    cap.release()
    writer.release()
    print(f"Đã lưu video kết quả tại: {output_path}")
