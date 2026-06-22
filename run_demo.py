import cv2
import yolo_detector

# Load the current best model
model = yolo_detector.load_model(r"runs\detect\motorbike_yolo11n\weights\best.pt")

# Read a test image
image_path = r"D:\1ComputerVisionProject1\motorbike.yolov11\valid\images\172-19-19-150_03_20230227105910150_jpg.rf.JIzzXC2w7FHelpgTDAG1.jpg"
frame = cv2.imread(image_path)

# Run detection
boxes = yolo_detector.detect_frame(model, frame, conf=0.25)
output = yolo_detector.draw_boxes(frame, boxes)

# Save the output
output_path = "demo_result.jpg"
cv2.imwrite(output_path, output)
print(f"Detection completed! Result saved to {output_path}")
