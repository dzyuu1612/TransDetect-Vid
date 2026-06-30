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

    def detect_frame(self, bgr_frame, conf_threshold=0.25, iou_threshold=0.45):
        if bgr_frame is None or bgr_frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        results = self.model.predict(
            bgr_frame,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.cpu()
            class_ids = boxes.cls.numpy().astype(int)
            xyxy = boxes.xyxy.numpy().astype(int)
            confidences = boxes.conf.numpy()

            for class_id, bbox, confidence in zip(class_ids, xyxy, confidences):
                class_name = str(self.model.names[class_id])
                normalized_name = class_name.strip().lower()
                if normalized_name not in self.VEHICLE_NAMES:
                    continue

                detections.append(
                    {
                        "bbox": bbox.tolist(),
                        "class": class_name,
                        "confidence": float(confidence),
                    }
                )

        return detections
