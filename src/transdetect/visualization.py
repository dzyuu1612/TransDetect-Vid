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


def draw_motion_vectors(frame, points, motion_vectors, display_scale=5.0):
    """Vẽ mũi tên chuyển động từ Lucas-Kanade.

    Chuyển động thực giữa hai khung hình liên tiếp thường chỉ vài pixel
    (video 25-30 FPS): vẽ đúng tỉ lệ khiến mũi tên bị chấm tròn bán kính 3px
    che khuất, gần như không thấy được, đặc biệt sau khi Dashboard thu nhỏ
    khung hình về 480p để hiển thị. `display_scale` chỉ phóng đại ĐỘ DÀI mũi
    tên khi VẼ để mắt người nhìn rõ hướng chuyển động; điểm p_curr (chấm
    tròn) vẫn ở đúng vị trí thật, và motion_vectors trả về cho lời gọi hàm
    không hề bị đổi - không có dữ liệu chuyển động nào được xuất ra CSV/JSON
    nên việc phóng đại này chỉ ảnh hưởng phần vẽ, không ảnh hưởng số liệu.
    """
    output = frame.copy()
    for (xc, yc), (u, v) in zip(points, motion_vectors):
        p_curr = (int(xc), int(yc))
        p_prev = (int(xc - u * display_scale), int(yc - v * display_scale))
        cv2.arrowedLine(output, p_prev, p_curr, (0, 0, 255), 2, tipLength=0.4)
        cv2.circle(output, p_curr, 3, (255, 0, 0), -1)
    return output
