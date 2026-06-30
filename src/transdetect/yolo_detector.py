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

class Yolo11VehicleDetector:
    def __init__(self, model_path="yolo11n.pt"):
        """Khởi tạo YOLO11 và giới hạn các lớp phương tiện trong MS COCO."""
        self.model = YOLO(model_path)
        import torch
        if torch.cuda.is_available():
            self.model.to("cuda:0")
            
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = {2, 3, 5, 7}

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
            if result.boxes is None or len(result.boxes) == 0:
                continue
                
            # Move all tensors to CPU and numpy at once to prevent GPU-CPU sync bottleneck
            boxes = result.boxes.cpu()
            cls_arr = boxes.cls.numpy()
            xyxy_arr = boxes.xyxy.numpy()
            conf_arr = boxes.conf.numpy()
            
            for i in range(len(boxes)):
                class_id = int(cls_arr[i])
                if class_id not in self.target_classes:
                    continue

                x1, y1, x2, y2 = map(int, xyxy_arr[i])
                detected_vehicles.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class": self.model.names[class_id],
                        "confidence": float(conf_arr[i]),
                    }
                )

        return detected_vehicles
