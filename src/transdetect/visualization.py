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


def draw_motion_vectors(frame, points, motion_vectors, display_scale=8.0,
                        min_arrow_length=10.0):
    """Vẽ mũi tên chuyển động từ Lucas-Kanade.

    Chuyển động thực giữa hai khung hình liên tiếp thường chỉ 1-4 pixel (còn
    ít hơn nữa với video quay ở FPS cao, ví dụ 59,94 FPS thực tế của
    duong_pho_sg(3).mp4 - khoảng cách thời gian giữa 2 frame ngắn hơn một
    nửa so với video 30 FPS cùng tốc độ xe thật). Vẽ đúng tỉ lệ 1:1 khiến mũi
    tên bị chấm tròn bán kính 3px che khuất gần hết, rồi Dashboard còn thu
    nhỏ khung hình về 480p để hiển thị (§3.7), làm nó nhỏ thêm lần nữa.

    `display_scale` phóng đại độ dài mũi tên khi VẼ; `min_arrow_length` đảm
    bảo NGAY CẢ khi chuyển động đo được rất nhỏ, mũi tên vẫn đủ dài để thấy
    (hướng vẫn giữ đúng, chỉ độ dài được kéo lên sàn tối thiểu). Điểm có
    chuyển động ĐÚNG BẰNG 0 (điểm tĩnh thật sự - nền, cột đèn...) vẫn chỉ vẽ
    chấm tròn, KHÔNG vẽ mũi tên giả - không bịa hướng cho thứ không di
    chuyển. Điểm p_curr (chấm tròn) luôn ở đúng vị trí thật, và
    motion_vectors trả về cho lời gọi hàm không hề bị đổi - không có dữ liệu
    chuyển động nào được xuất ra CSV/JSON nên việc phóng đại này chỉ ảnh
    hưởng phần vẽ, không ảnh hưởng số liệu báo cáo.
    """
    output = frame.copy()
    for (xc, yc), (u, v) in zip(points, motion_vectors):
        p_curr = (int(xc), int(yc))
        magnitude = (u ** 2 + v ** 2) ** 0.5
        if magnitude < 1e-3:
            # Chuyển động ~0: điểm tĩnh thật, không có hướng nào để vẽ.
            cv2.circle(output, p_curr, 3, (255, 0, 0), -1)
            continue

        draw_length = max(magnitude * display_scale, min_arrow_length)
        ux, uy = u / magnitude, v / magnitude  # vector đơn vị - giữ đúng hướng
        p_prev = (int(xc - ux * draw_length), int(yc - uy * draw_length))
        cv2.arrowedLine(output, p_prev, p_curr, (0, 0, 255), 2, tipLength=0.4)
        cv2.circle(output, p_curr, 3, (255, 0, 0), -1)
    return output
