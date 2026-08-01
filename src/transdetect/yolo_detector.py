"""Nhánh YOLO11, hỗ trợ model COCO và model custom có thứ tự lớp bất kỳ."""

import torch
from ultralytics import YOLO


class Yolo11VehicleDetector:
    VEHICLE_NAMES = {
        "car",
        "motorcycle",
        "motorbike",
        "bus",
        "truck",
        "container truck",
    }

    def __init__(self, model_path="yolo11n.pt"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect_frame(self, bgr_frame, conf_threshold=0.25, iou_threshold=0.45,
                     max_det=300, img_size=640):
        """Chạy YOLO11 trên một khung hình và trả về danh sách phương tiện.

        `img_size` là kích thước mà model nhận sau khi letterbox, KHÔNG phải độ
        phân giải của video nguồn. Mặc định 640 đúng bằng giá trị Ultralytics
        vẫn dùng ngầm trước đây, nên mọi nơi đang gọi hàm giữ nguyên hành vi;
        tham số này chỉ để evaluator truyền và ghi lại giá trị một cách tường
        minh trong metadata.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        results = self.model.predict(
            bgr_frame,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            imgsz=img_size,
            device=self.device,
            verbose=False,
        )
        detected_vehicles = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes.cpu():
                class_id = int(box.cls[0])
                class_name = str(self.model.names[class_id])
                if class_name.strip().lower() not in self.VEHICLE_NAMES:
                    continue

                detected_vehicles.append({
                    "bbox": box.xyxy[0].numpy().astype(int).tolist(),
                    "class": class_name,
                    "confidence": float(box.conf[0]),
                })

        return detected_vehicles
