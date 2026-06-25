"""
optical_flow.py
-----------------
Theo dõi chuyển động bằng Lucas-Kanade Optical Flow.

Tương ứng Listing 3.3 trong báo cáo (Mục 3.4.1). Đóng gói trạng thái điểm
đặc trưng trong một lớp. Khi tập điểm rỗng, Shi-Tomasi được gọi để khởi tạo;
các frame sau sử dụng Pyramidal Lucas-Kanade và chỉ giữ các điểm có status = 1.
"""

import cv2
import numpy as np


class LucasKanadeTracker:
    def __init__(self):
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )
        self.prev_points = None

    def reset(self):
        """Buộc bộ theo dõi khởi tạo lại điểm đặc trưng ở frame kế tiếp."""
        self.prev_points = None

    def track_features(self, prev_gray, curr_gray):
        """Trả về điểm hiện tại và vector v = p(t+1) - p(t)."""
        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_points = cv2.goodFeaturesToTrack(
                prev_gray, mask=None, **self.feature_params
            )

        if self.prev_points is None or len(self.prev_points) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            self.prev_points,
            None,
            **self.lk_params,
        )
        if curr_points is None or status is None:
            self.reset()
            return np.empty((0, 2)), np.empty((0, 2))

        valid = status.ravel() == 1
        valid_prev = self.prev_points.reshape(-1, 2)[valid]
        valid_curr = curr_points.reshape(-1, 2)[valid]
        motion_vectors = valid_curr - valid_prev

        self.prev_points = valid_curr.reshape(-1, 1, 2)
        return valid_curr, motion_vectors
