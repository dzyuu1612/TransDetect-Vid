"""
classical_detector.py
-----------------------
Phép biến đổi thủ công: Thresholding lặp + Sobel + trích bounding box.

Tương ứng Listing 3.2 trong báo cáo (Mục 3.3.1). Tách thành ba hàm độc lập:
ước lượng ngưỡng lặp, tính độ lớn gradient Sobel và trích vùng liên thông.
Bounding box trả về theo định dạng xyxy [x1, y1, x2, y2] để đồng nhất với YOLO.
"""

import cv2
import numpy as np


def iterative_global_threshold(gray_img, epsilon=1e-3, max_iter=100):
    """Ước lượng ngưỡng T bằng cách lặp trung bình hai lớp mức xám."""
    if gray_img.ndim != 2:
        raise ValueError("gray_img phải là ảnh xám một kênh.")

    threshold = float(np.mean(gray_img))
    for _ in range(max_iter):
        foreground = gray_img[gray_img > threshold]
        background = gray_img[gray_img <= threshold]

        # Nếu một lớp rỗng, không thể cập nhật trung bình hai lớp ổn định.
        if foreground.size == 0 or background.size == 0:
            break

        mean_foreground = float(np.mean(foreground))
        mean_background = float(np.mean(background))
        new_threshold = (mean_foreground + mean_background) / 2.0

        if abs(new_threshold - threshold) < epsilon:
            threshold = new_threshold
            break
        threshold = new_threshold

    binary_mask = np.where(gray_img > threshold, 255, 0).astype(np.uint8)
    return binary_mask, threshold


def sobel_edge_detection(gray_img, edge_threshold=50):
    """Tính ||grad I||2 từ đạo hàm Sobel theo hai trục x và y."""
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    _, edge_mask = cv2.threshold(
        magnitude, edge_threshold, 255, cv2.THRESH_BINARY
    )
    return edge_mask


def detect_vehicle_candidates(gray_img, min_area=500, max_aspect_ratio=4.0):
    """Kết hợp mask mức xám và mask biên để trích bounding box ứng viên."""
    binary_mask, threshold = iterative_global_threshold(gray_img)
    edge_mask = sobel_edge_detection(gray_img, edge_threshold=40)
    combined_mask = cv2.bitwise_or(binary_mask, edge_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(
        dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bounding_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / float(max(height, 1))
        if aspect_ratio <= max_aspect_ratio:
            bounding_boxes.append([x, y, x + width, y + height])
    counts = {
        "Car": len(bounding_boxes),
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0
    }
    return bounding_boxes, counts, combined_mask, threshold
