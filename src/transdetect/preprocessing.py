"""
preprocessing.py
-----------------
Module tiền xử lý ảnh cho nhánh truyền thống.

Tương ứng Listing 3.1 trong báo cáo (Mục 3.2.1). Hiện thực hai phép biến đổi
ở Mục 2.2: cân bằng histogram và lọc trung vị. Ảnh màu được đưa về ma trận
cường độ một kênh, sau đó cân bằng histogram và lọc trung vị.
"""

import cv2
import numpy as np


def histogram_equalization(gray_img: np.ndarray) -> np.ndarray:
    """Tăng độ tương phản của ảnh xám bằng cân bằng histogram."""
    if gray_img.ndim != 2:
        raise ValueError("gray_img phải là ảnh xám một kênh.")
    return cv2.equalizeHist(gray_img)


def median_filter(gray_img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Khử nhiễu muối tiêu, đồng thời bảo toàn biên tốt hơn lọc trung bình."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size phải là số lẻ và không nhỏ hơn 3.")
    return cv2.medianBlur(gray_img, kernel_size)


def preprocess_frame(bgr_frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """BGR -> grayscale -> histogram equalization -> median filter."""
    if bgr_frame is None or bgr_frame.size == 0:
        raise ValueError("Khung hình đầu vào rỗng.")

    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    equalized = histogram_equalization(gray)
    denoised = median_filter(equalized, kernel_size)
    return denoised
