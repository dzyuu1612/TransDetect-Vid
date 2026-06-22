"""
yolo_detector.py
------------------
Chay YOLO11 tren video - giai thich CHI TIET TUNG DONG.
"""

from ultralytics import YOLO


def load_model(model_path: str = "yolo11n.pt"):
    """Tai model YOLO11."""
    # YOLO(model_path): vua dung de load model GOC (yolo11n.pt, tu dong tai ve)
    # vua dung de load model DA TU TRAIN (vi du runs/detect/.../best.pt)
    model = YOLO(model_path)
    return model


def detect_frame(model, frame, conf: float = 0.25):
    """Chay YOLO11 tren 1 frame."""
    # model.predict: chay mo hinh tren 1 anh/frame
    # conf=0.25: chi giu lai box co do tin cay (confidence) >= 0.25 (25%)
    # verbose=False: tat log chi tiet in ra man hinh moi lan predict (cho gon)
    results = model.predict(frame, conf=conf, verbose=False)

    boxes = []                        # danh sach rong de chua ket qua dang de doc
    for r in results:                 # results la list (thuong chi co 1 phan tu vi chi dua 1 frame)
        for box in r.boxes:           # r.boxes chua tat ca box phat hien duoc trong frame nay
            # box.xyxy[0]: toa do box dang (x1,y1,x2,y2) - goc trai-tren va goc phai-duoi
            # .tolist(): doi tu tensor (kieu du lieu cua pytorch) sang list so thuong cua python
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_id = int(box.cls[0])           # box.cls la ma so class (vi du 0,1,2...)
            class_name = model.names[class_id]   # model.names la tu dien {ma_so: ten_class}
            confidence = float(box.conf[0])      # do tin cay cua du doan nay (0.0 - 1.0)

            # them 1 tuple ket qua vao danh sach, da doi sang kieu du lieu python thuong (int/float/str)
            boxes.append((int(x1), int(y1), int(x2), int(y2), class_name, confidence))

    return boxes


def draw_boxes(frame, boxes):
    """Ve box + nhan + confidence len anh."""
    import cv2   # import o day (khong o dau file) de file nay van chay duoc khi
                  # may chi can train model va chua can cv2 cho viec hien thi

    output = frame.copy()             # copy anh, tranh ve de len bien frame goc dang dung o noi khac
    for (x1, y1, x2, y2, class_name, confidence) in boxes:
        # ve khung chu nhat tu (x1,y1) den (x2,y2), mau xanh la (0,255,0), do day 2
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name} {confidence:.2f}"   # vi du: "motorbike 0.87"
        # ve chu len anh, vi tri ngay tren goc trai box (lui len 8 pixel, khong de am)
        cv2.putText(output, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output


if __name__ == "__main__":
    import sys
    import cv2

    if len(sys.argv) < 2:
        print("Cach dung: python yolo_detector.py duong_dan_video.mp4 [duong_dan_model.pt]")
        sys.exit(1)

    video_path = sys.argv[1]
    # neu nguoi dung khong truyen model, dung model goc pre-trained tren COCO
    model_path = sys.argv[2] if len(sys.argv) > 2 else "yolo11n.pt"

    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)   # mo file video de doc tung frame

    while True:
        ret, frame = cap.read()    # doc 1 frame; ret=False khi het video
        if not ret:
            break

        boxes = detect_frame(model, frame)    # chay YOLO11 tren frame nay
        output = draw_boxes(frame, boxes)     # ve ket qua len anh

        cv2.imshow("YOLO11 Detection", output)
        # waitKey(30) & 0xFF: doc ma phim nguoi dung bam (cho 30ms);
        # neu la phim 'q' thi dung vong lap
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
