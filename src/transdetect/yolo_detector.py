"""
yolo_detector.py
------------------
Nhánh học sâu YOLO11 (phép biến đổi học được).

Tương ứng Listing 3.4 trong báo cáo (Mục 3.5.1). Cung cấp giao diện thống nhất:
đầu vào là một frame BGR, đầu ra là danh sách bounding box, nhãn và độ tin cậy.

Lọc lớp theo TÊN phương tiện để hoạt động đúng cho cả model COCO pre-trained
lẫn model tự huấn luyện (data.yaml có thứ tự lớp khác COCO). Tập COCO ID
{2: car, 3: motorcycle, 5: bus, 7: truck} được giữ làm phương án dự phòng khi
model không cung cấp tên lớp.
"""

from ultralytics import YOLO


# Tên các lớp phương tiện cần giữ (chuẩn hóa chữ thường).
VEHICLE_NAMES = {"car", "motorcycle", "motorbike", "bus", "truck"}

# Phương án dự phòng theo COCO ID khi không tra được tên lớp.
COCO_VEHICLE_IDS = {2, 3, 5, 7}


class Yolo11VehicleDetector:
    def __init__(self, model_path="yolo11n.pt"):
        """Khởi tạo YOLO11 và xác định tập lớp phương tiện cần giữ."""
        self.model = YOLO(model_path)
        # Tập COCO ID dùng khi model trả về đúng bộ nhãn MS COCO.
        self.target_classes = COCO_VEHICLE_IDS

    def _is_vehicle(self, class_id):
        """Quyết định một lớp có phải phương tiện không, ưu tiên theo tên."""
        names = getattr(self.model, "names", None)
        if names and class_id in names:
            return names[class_id].lower() in VEHICLE_NAMES
        return class_id in self.target_classes

    def detect_frame(self, bgr_frame, conf_threshold=0.25, iou_threshold=0.45):
        """Suy luận một frame và chuẩn hóa kết quả thành danh sách từ điển."""
        results = self.model.predict(
            bgr_frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        detected_vehicles = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if not self._is_vehicle(class_id):
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detected_vehicles.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class": self.model.names[class_id],
                        "confidence": float(box.conf[0]),
                    }
                )

        return detected_vehicles
