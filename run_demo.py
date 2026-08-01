import os

import cv2
import yolo_detector

# Load the current best model
model = yolo_detector.load_model(r"runs\detect\motorbike_yolo11n\weights\best.pt")

# Read a test image
image_path = os.environ.get(
    "TRANSDETECT_DEMO_IMAGE",
    "datasets/Vehicle Detection.v7i.yolov11/valid/images/sample.jpg",
)
frame = cv2.imread(image_path)

# Run detection
boxes = yolo_detector.detect_frame(model, frame, conf=0.25)
output = yolo_detector.draw_boxes(frame, boxes)

# Save the output
output_path = "demo_result.jpg"
cv2.imwrite(output_path, output)
print(f"Detection completed! Result saved to {output_path}")
