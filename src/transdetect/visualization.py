"""
visualization.py
-------------------
Vẽ kết quả của hai nhánh lên frame: bounding box, nhãn lớp, vector chuyển động.

Khối hiển thị này tách khỏi khối phát hiện để không phụ thuộc trực tiếp vào kiểu
dữ liệu nội bộ của Ultralytics (xem Mục 3.5.1).
"""

import cv2

from . import config


def draw_classical_boxes(frame, boxes):
    """Vẽ bounding box xyxy của nhánh truyền thống."""
    output = frame.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            output, (x1, y1), (x2, y2), config.BOX_COLOR, config.BOX_THICKNESS
        )
    return output


def draw_yolo_detections(frame, detections):
    """Vẽ box + nhãn + confidence từ danh sách từ điển của YOLO11."""
    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(
            output, (x1, y1), (x2, y2), config.BOX_COLOR, config.BOX_THICKNESS
        )
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            config.BOX_COLOR,
            2,
        )
    return output


def draw_motion_vectors(frame, points, motion_vectors):
    """Vẽ mũi tên chuyển động từ Lucas-Kanade."""
    output = frame.copy()
    for (xc, yc), (u, v) in zip(points, motion_vectors):
        p_curr = (int(xc), int(yc))
        p_prev = (int(xc - u), int(yc - v))
        cv2.arrowedLine(output, p_prev, p_curr, (0, 0, 255), 2, tipLength=0.4)
        cv2.circle(output, p_curr, 3, (255, 0, 0), -1)
    return output
