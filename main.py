"""
main.py
---------
File chay chinh - giai thich CHI TIET TUNG DONG.
"""

import argparse   # thu vien doc tham so dong lenh (--input, --method, ...)
import cv2
import numpy as np

import preprocessing       # file preprocessing.py cung thu muc
import classical_detector  # file classical_detector.py cung thu muc
import optical_flow        # file optical_flow.py cung thu muc
import yolo_detector       # file yolo_detector.py cung thu muc
import train_yolo          # file train_yolo.py cung thu muc


def run_classical(input_path: str, output_path: str):
    """Chay nhanh truyen thong tren toan bo video."""
    cap = cv2.VideoCapture(input_path)              # mo video dau vao de doc tung frame
    fps = cap.get(cv2.CAP_PROP_FPS) or 25            # lay FPS goc cua video; neu khong doc duoc, dung 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # chieu rong frame (pixel)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # chieu cao frame (pixel)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ma "codec" (kieu nen video) dung de luu file .mp4
    # VideoWriter: doi tuong dung de GHI tung frame ra thanh 1 file video moi
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_gray = None     # frame xam truoc do, dung cho Lucas-Kanade (luc dau chua co nen = None)
    prev_points = None   # diem dang theo doi, luc dau chua co nen = None

    while True:
        ret, frame = cap.read()    # doc 1 frame; ret=False khi het video
        if not ret:
            break                   # het video -> thoat vong lap

        # --- Buoc 1: tien xu ly (xam + can bang histogram + khu nhieu) ---
        pre = preprocessing.preprocess(frame)

        # --- Buoc 2: Threshold + Sobel de tim vung ung vien ---
        boxes, _ = classical_detector.detect(pre)   # dau "_" nghia la khong dung gia tri mask tra ve o day

        output = frame.copy()       # se ve ket qua len ban sao cua frame goc
        for (x, y, w, h) in boxes:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # ve tung box mau xanh la

        # --- Buoc 3: Lucas-Kanade theo doi chuyen dong ---
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # doi frame hien tai ve anh xam
        if prev_gray is not None:    # chi theo doi duoc tu frame THU 2 tro di (can co frame truoc)
            if prev_points is None or len(prev_points) < 5:
                # neu chua co diem hoac con qua it diem, tim lai diem moi de theo doi
                prev_points = optical_flow.get_good_features(prev_gray)

            if prev_points is not None:
                good_prev, good_curr = optical_flow.track_points(
                    prev_gray, curr_gray, prev_points
                )
                output = optical_flow.draw_motion_vectors(output, good_prev, good_curr)
                # cap nhat danh sach diem theo doi cho vong lap (frame) ke tiep
                prev_points = good_curr.reshape(-1, 1, 2).astype(np.float32)

        prev_gray = curr_gray    # frame hien tai thanh "frame truoc" cho vong lap sau

        writer.write(output)             # ghi frame ket qua vao file video output
        cv2.imshow("Pipeline truyen thong", output)   # hien thi truc tiep len man hinh
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   # bam 'q' de dung som

    cap.release()             # dong file video dau vao
    writer.release()          # luu va dong file video dau ra
    cv2.destroyAllWindows()   # dong cac cua so hien thi
    print(f"Da luu video ket qua tai: {output_path}")


def run_yolo(input_path: str, output_path: str, model_path: str, conf: float):
    """Chay nhanh YOLO11 tren toan bo video."""

    model = yolo_detector.load_model(model_path)   # tai model YOLO11 (goc hoac da tu train)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = yolo_detector.detect_frame(model, frame, conf=conf)  # chay YOLO11 tren frame
        output = yolo_detector.draw_boxes(frame, boxes)              # ve box+nhan len anh

        writer.write(output)
        cv2.imshow("YOLO11", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Da luu video ket qua tai: {output_path}")


if __name__ == "__main__":
    # Tao bo doc tham so dong lenh
    parser = argparse.ArgumentParser(description="TransDetect-Vid: chay pipeline phat hien doi tuong trong video.")
    parser.add_argument("--input", required=True, help="Duong dan video dau vao")
    parser.add_argument("--output", required=True, help="Duong dan video ket qua dau ra")
    parser.add_argument("--method", choices=["classical", "yolo"], required=True,
                         help="Chon nhanh xu ly: classical hoac yolo")
    parser.add_argument("--model", default="yolo11n.pt",
                         help="Duong dan model YOLO11 (chi dung khi --method yolo)")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="Nguong confidence cho YOLO11 (chi dung khi --method yolo)")

    args = parser.parse_args()   # doc thuc te cac tham so nguoi dung go vao dong lenh

    if args.method == "classical":
        run_classical(args.input, args.output)
    else:
        run_yolo(args.input, args.output, args.model, args.conf)
