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


def draw_motion_vectors(frame, points, motion_vectors, min_motion=1.0,
                        arrow_length=30.0):
    """Vẽ điểm đặc trưng và mũi tên chuyển động từ Lucas-Kanade.

    Quy tắc vẽ rất đơn giản, chỉ có hai trường hợp:

    1. Điểm gần như đứng yên (độ dời < `min_motion` pixel) -> chỉ vẽ CHẤM
       XANH. Đây là các điểm bám trên nền tĩnh: nhà, cây, biển hiệu. Đo trên
       video sg(3) thật, khoảng 80/100 điểm rơi vào nhóm này với độ dời
       trung bình chỉ 0,67 px - tức là nhiễu dưới một pixel, không phải
       chuyển động thật. Không vẽ mũi tên cho chúng để tránh biến nhiễu
       thành tín hiệu giả.

    2. Điểm di chuyển thật (độ dời >= `min_motion`) -> vẽ thêm MŨI TÊN ĐỎ
       chỉ đúng hướng đi. Đây là các điểm bám trên xe đang chạy, khoảng
       20/100 điểm, độ dời 1-6 px mỗi frame.

    Vì sao phải vẽ mũi tên dài `arrow_length` cố định thay vì đúng độ dời
    thật: độ dời thật chỉ 1-6 px, trong khi chấm tròn đã có bán kính 3 px và
    Dashboard còn thu nhỏ khung hình 720p xuống 480p trước khi hiển thị
    (Mục 3.7). Vẽ đúng tỉ lệ 1:1 thì mũi tên bị chấm tròn nuốt mất, người
    xem không thấy gì. Mũi tên ở đây thể hiện HƯỚNG di chuyển (luôn chính
    xác), còn độ dài chỉ để nhìn thấy được.

    Giá trị `motion_vectors` truyền vào không hề bị sửa, và không có dữ liệu
    chuyển động nào được xuất ra CSV/JSON, nên cách vẽ này không ảnh hưởng
    tới bất kỳ số liệu nào trong báo cáo.
    """
    output = frame.copy()
    for (xc, yc), (u, v) in zip(points, motion_vectors):
        p_curr = (int(xc), int(yc))
        do_doi = (u ** 2 + v ** 2) ** 0.5

        if do_doi >= min_motion:
            # Chia cho do_doi để được vector đơn vị -> giữ nguyên hướng,
            # rồi nhân với arrow_length để mũi tên đủ dài mà nhìn thấy.
            huong_x, huong_y = u / do_doi, v / do_doi
            p_prev = (int(xc - huong_x * arrow_length),
                      int(yc - huong_y * arrow_length))
            cv2.arrowedLine(output, p_prev, p_curr, (0, 0, 255), 3,
                            tipLength=0.35)

        cv2.circle(output, p_curr, 3, (255, 0, 0), -1)
    return output
