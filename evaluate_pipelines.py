"""
evaluate_pipelines.py
-----------------------
Đánh giá định lượng, so sánh nhánh truyền thống và YOLO11 trên CÙNG một tập
frame có ground truth.

Script này CHỈ ĐỌC các module phát hiện hiện có, không sửa thuật toán:
    - preprocessing.preprocess_frame
    - classical_detector.detect_vehicle_candidates
    - Yolo11VehicleDetector.detect_frame

Nguyên tắc đánh giá (xem Mục 4.2.2 của báo cáo):
    - So sánh CLASS-AGNOSTIC: gộp Car/Motorcycle/Bus/Truck thành một lớp
      "vehicle". Bắt buộc phải như vậy vì nhánh truyền thống chỉ sinh vùng ứng
      viên, không phân loại được loại xe — nếu so sánh có phân biệt lớp thì
      nhánh truyền thống sẽ luôn sai 100%, không phản ánh đúng khả năng ĐỊNH VỊ
      phương tiện của nó.
    - Ghép prediction với ground truth MỘT-MỘT theo IoU giảm dần, ngưỡng
      IoU >= 0.5 mới tính là True Positive.
    - Không dùng confidence của YOLO làm độ chính xác; confidence chỉ là điểm
      tin cậy của dự đoán, không phải thước đo đúng/sai.
    - Không tính mAP cho nhánh truyền thống: mAP cần confidence score để xếp
      hạng prediction và dựng đường Precision-Recall, mà nhánh truyền thống
      không có confidence chuẩn.

Cách chạy:
    python evaluate_pipelines.py --images evaluation/images \
        --labels evaluation/labels --model yolo11n.pt --conf 0.25 \
        --iou-match 0.5 --output evaluation/results

Chạy self-test cho các hàm IoU / matching / metric (không cần dữ liệu):
    python evaluate_pipelines.py --selftest
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

from src.transdetect import classical_detector
from src.transdetect import config
from src.transdetect import preprocessing
from src.transdetect.yolo_detector import Yolo11VehicleDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def calculate_iou(box_a, box_b):
    """Tính IoU giữa hai box định dạng xyxy.

    IoU = diện tích phần giao / diện tích phần hợp. Trả về 0.0 khi hai box
    không giao nhau hoặc khi phần hợp bằng 0 (box suy biến).
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    # max(0, ...) xử lý trường hợp hai box KHÔNG giao nhau: khi đó
    # intersection_x2 < intersection_x1 nên hiệu âm, phải kẹp về 0.
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def load_yolo_ground_truth(label_path, image_width, image_height):
    """Đọc nhãn định dạng YOLO và đổi toạ độ chuẩn hoá sang xyxy pixel.

    Định dạng mỗi dòng: class_id x_center y_center width height, trong đó
    bốn giá trị toạ độ đã được chuẩn hoá về [0, 1] theo kích thước ảnh.

    class_id bị BỎ QUA có chủ đích: phép đánh giá là class-agnostic, mọi
    phương tiện đều được gộp thành một lớp "vehicle".
    """
    ground_truth_boxes = []

    if not label_path.exists():
        return ground_truth_boxes

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if len(values) < 5:
                continue

            _, center_x, center_y, box_width, box_height = values[:5]
            center_x = float(center_x) * image_width
            center_y = float(center_y) * image_height
            box_width = float(box_width) * image_width
            box_height = float(box_height) * image_height

            x1 = center_x - box_width / 2
            y1 = center_y - box_height / 2
            x2 = center_x + box_width / 2
            y2 = center_y + box_height / 2

            ground_truth_boxes.append([x1, y1, x2, y2])

    return ground_truth_boxes


def match_predictions(predicted_boxes, ground_truth_boxes, iou_threshold=0.5):
    """Ghép một-một prediction với ground truth theo IoU giảm dần.

    Thuật toán tham lam: xét mọi cặp (prediction, ground truth), sắp theo IoU
    giảm dần, rồi lần lượt chốt cặp có IoU cao nhất còn khả dụng. Mỗi
    prediction và mỗi ground truth chỉ được dùng đúng một lần — nhờ vậy hai
    prediction chồng lên cùng một chiếc xe chỉ được tính đúng một TP, còn
    prediction thừa thành FP.

    Trả về (true_positive, false_positive, false_negative).
    """
    candidate_pairs = []

    for prediction_index, predicted_box in enumerate(predicted_boxes):
        for ground_truth_index, ground_truth_box in enumerate(ground_truth_boxes):
            iou = calculate_iou(predicted_box, ground_truth_box)
            candidate_pairs.append((iou, prediction_index, ground_truth_index))

    candidate_pairs.sort(key=lambda item: item[0], reverse=True)

    matched_predictions = set()
    matched_ground_truths = set()
    true_positive = 0

    for iou, prediction_index, ground_truth_index in candidate_pairs:
        # Danh sách đã sắp giảm dần nên gặp cặp dưới ngưỡng là dừng được ngay,
        # mọi cặp phía sau chắc chắn cũng dưới ngưỡng.
        if iou < iou_threshold:
            break

        if prediction_index in matched_predictions:
            continue

        if ground_truth_index in matched_ground_truths:
            continue

        matched_predictions.add(prediction_index)
        matched_ground_truths.add(ground_truth_index)
        true_positive += 1

    false_positive = len(predicted_boxes) - true_positive
    false_negative = len(ground_truth_boxes) - true_positive

    return true_positive, false_positive, false_negative


def calculate_metrics(true_positive, false_positive, false_negative):
    """Tính Precision, Recall và F1 từ TP/FP/FN.

    Precision = TP / (TP + FP): trong các box phát hiện, bao nhiêu box đúng.
    Recall    = TP / (TP + FN): trong các xe thật, tìm được bao nhiêu.
    F1        = trung bình điều hoà của Precision và Recall.

    Mọi mẫu số bằng 0 đều trả về 0.0 thay vì lỗi chia cho 0 (ví dụ khi một
    phương pháp không phát hiện được gì trên toàn tập).
    """
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1


def detect_with_classical(frame):
    """Chạy nhánh truyền thống với ĐÚNG tham số trong config.py.

    Lucas-Kanade không tham gia ở đây: trong mã nguồn hiện tại nó chỉ vẽ điểm
    và vector chuyển động, không sửa và không lọc bounding box, nên không ảnh
    hưởng tới kết quả phát hiện được đánh giá.
    """
    preprocessed = preprocessing.preprocess_frame(
        frame,
        config.MEDIAN_KERNEL_SIZE,
    )

    boxes, _, _ = classical_detector.detect_vehicle_candidates(
        preprocessed,
        min_area=config.MIN_CONTOUR_AREA,
        max_area=config.MAX_CONTOUR_AREA,
        max_aspect_ratio=config.MAX_ASPECT_RATIO,
        edge_threshold=config.SOBEL_EDGE_THRESHOLD,
    )

    return boxes


def detect_with_yolo(detector, frame, conf_threshold):
    """Chạy YOLO11 và chỉ lấy toạ độ bounding box.

    Nhãn lớp bị bỏ đi có chủ đích để phép so sánh là class-agnostic; confidence
    cũng không được dùng vào bất kỳ chỉ số nào.
    """
    detections = detector.detect_frame(
        frame,
        conf_threshold=conf_threshold,
        iou_threshold=config.IOU_THRESHOLD,
        max_det=config.MAX_DET,
    )

    return [detection["bbox"] for detection in detections]


def evaluate_method(
    method_name,
    image_paths,
    label_directory,
    prediction_function,
    iou_threshold,
):
    """Đánh giá một phương pháp trên toàn bộ tập ảnh.

    TP/FP/FN được cộng dồn trên toàn tập rồi mới tính Precision/Recall/F1
    (micro-average), thay vì lấy trung bình chỉ số của từng frame — cách này
    tránh việc một frame chỉ có 1 xe lại có cùng trọng số với frame có 30 xe.
    """
    total_ground_truth = 0
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    frame_results = []
    missing_label_files = []

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  Bỏ qua ảnh không đọc được: {image_path}")
            continue

        image_height, image_width = frame.shape[:2]
        label_path = label_directory / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_label_files.append(image_path.name)

        ground_truth_boxes = load_yolo_ground_truth(
            label_path,
            image_width,
            image_height,
        )
        predicted_boxes = prediction_function(frame)

        true_positive, false_positive, false_negative = match_predictions(
            predicted_boxes,
            ground_truth_boxes,
            iou_threshold=iou_threshold,
        )

        total_ground_truth += len(ground_truth_boxes)
        total_true_positive += true_positive
        total_false_positive += false_positive
        total_false_negative += false_negative

        frame_results.append({
            "method": method_name,
            "image": image_path.name,
            "ground_truth": len(ground_truth_boxes),
            "predictions": len(predicted_boxes),
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
        })

    if missing_label_files:
        print(
            f"  CẢNH BÁO [{method_name}]: {len(missing_label_files)} ảnh không "
            f"có file nhãn tương ứng, bị coi như 0 ground truth. "
            f"Ví dụ: {missing_label_files[:3]}"
        )

    precision, recall, f1 = calculate_metrics(
        total_true_positive,
        total_false_positive,
        total_false_negative,
    )

    summary = {
        "method": method_name,
        "number_of_images": len(frame_results),
        "number_of_ground_truth_boxes": total_ground_truth,
        "tp": total_true_positive,
        "fp": total_false_positive,
        "fn": total_false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou_threshold": iou_threshold,
    }

    return summary, frame_results


def write_csv(output_path, rows):
    """Ghi danh sách dictionary ra CSV.

    Dùng encoding utf-8-sig để Excel trên Windows mở đúng tiếng Việt.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_selftest():
    """Kiểm thử các hàm thuần tính toán: IoU, matching và metric.

    Không cần ảnh, nhãn hay model — chạy được ngay cả khi chưa có ground truth.
    """
    failures = []

    def check(name, actual, expected, tolerance=1e-6):
        ok = abs(actual - expected) <= tolerance
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: được {actual}, mong đợi {expected}")
        if not ok:
            failures.append(name)

    print("1) calculate_iou")
    # Hai box trùng khít hoàn toàn -> IoU = 1
    check("box trùng khít", calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
    # Hai box rời nhau -> IoU = 0
    check("box rời nhau", calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
    # Giao 50x100=5000, hợp = 10000+10000-5000 = 15000 -> 1/3
    check(
        "chồng một nửa",
        calculate_iou([0, 0, 100, 100], [50, 0, 150, 100]),
        5000 / 15000,
    )
    # Box suy biến (diện tích 0) không được gây lỗi chia cho 0
    check("box suy biến", calculate_iou([5, 5, 5, 5], [5, 5, 5, 5]), 0.0)
    # Ví dụ trong báo cáo Mục 2.6.5: diện tích 100 và 120, giao 80
    # -> IoU = 80 / (100 + 120 - 80) = 0.5714...
    check(
        "ví dụ Mục 2.6.5",
        calculate_iou([0, 0, 10, 10], [2, 0, 14, 10]),
        80 / (100 + 120 - 80),
        tolerance=1e-4,
    )

    print("2) match_predictions")
    gt = [[0, 0, 10, 10], [100, 100, 110, 110]]

    # Dự đoán khớp hoàn hảo cả hai -> 2 TP, 0 FP, 0 FN
    tp, fp, fn = match_predictions(list(gt), gt, 0.5)
    check("khớp hoàn hảo TP", tp, 2)
    check("khớp hoàn hảo FP", fp, 0)
    check("khớp hoàn hảo FN", fn, 0)

    # Không có prediction nào -> toàn bộ ground truth thành FN
    tp, fp, fn = match_predictions([], gt, 0.5)
    check("không dự đoán TP", tp, 0)
    check("không dự đoán FN", fn, 2)

    # Dự đoán nhưng không có ground truth -> toàn bộ thành FP
    tp, fp, fn = match_predictions([[0, 0, 10, 10]], [], 0.5)
    check("không ground truth FP", fp, 1)

    # HAI prediction cùng chồng lên MỘT ground truth: chỉ 1 TP, 1 FP.
    # Đây là trường hợp quan trọng nhất - chứng minh ghép đúng một-một.
    tp, fp, fn = match_predictions(
        [[0, 0, 10, 10], [1, 1, 11, 11]],
        [[0, 0, 10, 10]],
        0.5,
    )
    check("hai pred một GT -> TP", tp, 1)
    check("hai pred một GT -> FP", fp, 1)
    check("hai pred một GT -> FN", fn, 0)

    # Chồng lấp dưới ngưỡng 0.5 -> không tính TP
    tp, fp, fn = match_predictions([[0, 0, 100, 100]], [[80, 0, 180, 100]], 0.5)
    check("dưới ngưỡng IoU -> TP", tp, 0)
    check("dưới ngưỡng IoU -> FP", fp, 1)
    check("dưới ngưỡng IoU -> FN", fn, 1)

    print("3) calculate_metrics")
    # Ví dụ tính tay trong tài liệu: TP=80, FP=20, FN=40
    precision, recall, f1 = calculate_metrics(80, 20, 40)
    check("Precision", precision, 0.80)
    check("Recall", recall, 80 / 120, tolerance=1e-4)
    check("F1", f1, 2 * 0.8 * (80 / 120) / (0.8 + 80 / 120), tolerance=1e-4)

    # Không phát hiện được gì -> mọi chỉ số bằng 0, không lỗi chia cho 0
    precision, recall, f1 = calculate_metrics(0, 0, 10)
    check("toàn FN Precision", precision, 0.0)
    check("toàn FN Recall", recall, 0.0)
    check("toàn FN F1", f1, 0.0)

    print()
    if failures:
        print(f"KẾT QUẢ: THẤT BẠI {len(failures)} kiểm thử: {failures}")
        return 1

    print("KẾT QUẢ: TẤT CẢ KIỂM THỬ ĐỀU PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Đánh giá Classical Pipeline và YOLO11 trên cùng ground truth "
            "(class-agnostic, IoU >= 0.5)."
        )
    )
    parser.add_argument(
        "--images",
        default="evaluation/images",
        help="Thư mục ảnh đánh giá.",
    )
    parser.add_argument(
        "--labels",
        default="evaluation/labels",
        help="Thư mục nhãn định dạng YOLO.",
    )
    parser.add_argument(
        "--model",
        default=config.DEFAULT_YOLO_MODEL,
        help="Đường dẫn model YOLO.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONF_THRESHOLD,
        help="Confidence threshold của YOLO.",
    )
    parser.add_argument(
        "--iou-match",
        type=float,
        default=0.5,
        help="Ngưỡng IoU để xác định True Positive.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results",
        help="Thư mục lưu kết quả.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Chạy kiểm thử các hàm IoU/matching/metric rồi thoát.",
    )
    args = parser.parse_args()

    if args.selftest:
        return run_selftest()

    image_directory = Path(args.images)
    label_directory = Path(args.labels)
    output_directory = Path(args.output)

    if not image_directory.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh: {image_directory}\n"
            f"Xem Mục 6 của BO_SUNG_DANH_GIA_DO_CHINH_XAC_TRANDETECT_VID.md "
            f"để biết cách chuẩn bị tập ground truth."
        )

    image_paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh đánh giá trong: {image_directory}\n"
            f"Cần trích xuất frame và gán nhãn trước khi chạy đánh giá."
        )

    label_paths = (
        sorted(label_directory.glob("*.txt")) if label_directory.is_dir() else []
    )
    print(f"Số ảnh: {len(image_paths)} | Số file nhãn: {len(label_paths)}")
    if len(image_paths) != len(label_paths):
        print(
            "  CẢNH BÁO: số ảnh và số file nhãn KHÔNG bằng nhau. Ảnh thiếu "
            "nhãn sẽ bị coi như không có phương tiện nào, làm Precision giảm "
            "sai lệch. Hãy kiểm tra lại trước khi đưa số vào báo cáo."
        )
    print()

    yolo_detector = Yolo11VehicleDetector(args.model)

    print("Đang đánh giá nhánh truyền thống...")
    classical_summary, classical_frames = evaluate_method(
        method_name="Classical",
        image_paths=image_paths,
        label_directory=label_directory,
        prediction_function=detect_with_classical,
        iou_threshold=args.iou_match,
    )

    print("Đang đánh giá YOLO11...")
    yolo_summary, yolo_frames = evaluate_method(
        method_name="YOLO11",
        image_paths=image_paths,
        label_directory=label_directory,
        prediction_function=lambda frame: detect_with_yolo(
            yolo_detector,
            frame,
            args.conf,
        ),
        iou_threshold=args.iou_match,
    )

    write_csv(
        output_directory / "summary_metrics.csv",
        [classical_summary, yolo_summary],
    )
    write_csv(
        output_directory / "per_frame_metrics.csv",
        classical_frames + yolo_frames,
    )

    print()
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 72)
    print(f"Ngưỡng ghép IoU: {args.iou_match} | YOLO conf: {args.conf}")
    print("So sánh class-agnostic (gộp Car/Motorcycle/Bus/Truck -> vehicle)")
    print("-" * 72)

    for summary in (classical_summary, yolo_summary):
        print(
            f"{summary['method']:<12} | "
            f"TP={summary['tp']:<5} "
            f"FP={summary['fp']:<5} "
            f"FN={summary['fn']:<5} | "
            f"P={summary['precision']:.4f} "
            f"R={summary['recall']:.4f} "
            f"F1={summary['f1']:.4f}"
        )

    print("=" * 72)
    print(f"Đã lưu kết quả tại: {output_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
