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
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

from src.transdetect import classical_detector
from src.transdetect import config
from src.transdetect import preprocessing
from src.transdetect.yolo_detector import Yolo11VehicleDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Bốn lớp phương tiện được phép trong ground truth, khớp evaluation/classes.txt.
# class_id chỉ dùng để KIỂM TRA tính hợp lệ của nhãn; khi chấm điểm thì bị bỏ
# qua vì phép đánh giá là class-agnostic (xem giải thích ở đầu file).
ALLOWED_CLASS_IDS = {0, 1, 2, 3}
CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}


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


def validate_evaluation_dataset(image_directory, label_directory,
                                allowed_class_ids=None):
    """Kiểm tra nghiêm ngặt tính hợp lệ của tập ảnh + nhãn TRƯỚC khi chấm điểm.

    Vì sao phải dừng hẳn thay vì cảnh báo rồi chạy tiếp: một ảnh thiếu file
    nhãn sẽ bị hiểu là "frame này không có phương tiện nào", nên mọi box mà
    hai pipeline phát hiện trên ảnh đó đều bị tính là False Positive. Chỉ cần
    vài ảnh thiếu nhãn là Precision đã sai lệch đáng kể, mà không có dấu hiệu
    gì trong file CSV kết quả. Sai lệch âm thầm kiểu này nguy hiểm hơn nhiều
    so với việc script báo lỗi và không chạy.

    Trả về dict thống kê. Ném ValueError kèm danh sách lỗi cụ thể nếu dataset
    không hợp lệ.
    """
    if allowed_class_ids is None:
        allowed_class_ids = ALLOWED_CLASS_IDS

    image_directory = Path(image_directory)
    label_directory = Path(label_directory)
    errors = []

    # --- 1 & 2: hai thư mục phải tồn tại ---
    if not image_directory.is_dir():
        raise ValueError(f"Không tìm thấy thư mục ảnh: {image_directory}")
    if not label_directory.is_dir():
        raise ValueError(f"Không tìm thấy thư mục nhãn: {label_directory}")

    image_paths = sorted(
        path for path in image_directory.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(
            f"Không có ảnh nào trong {image_directory}.\n"
            f"Chạy prepare_evaluation_frames.py để trích frame trước."
        )

    label_paths = sorted(label_directory.glob("*.txt"))
    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}

    # --- 3: mọi ảnh phải có đúng một file nhãn cùng stem ---
    missing_labels = sorted(image_stems - label_stems)
    if missing_labels:
        errors.append(
            f"{len(missing_labels)} ảnh KHÔNG có file nhãn tương ứng: "
            f"{missing_labels[:5]}{' ...' if len(missing_labels) > 5 else ''}"
        )

    # --- 4: không có file nhãn thừa (nhãn mà không có ảnh) ---
    orphan_labels = sorted(label_stems - image_stems)
    if orphan_labels:
        errors.append(
            f"{len(orphan_labels)} file nhãn KHÔNG có ảnh tương ứng: "
            f"{orphan_labels[:5]}{' ...' if len(orphan_labels) > 5 else ''}"
        )

    total_boxes = 0
    empty_label_files = 0
    class_counter = {class_id: 0 for class_id in allowed_class_ids}

    for image_path in image_paths:
        # --- 5: ảnh phải đọc được bằng OpenCV ---
        frame = cv2.imread(str(image_path))
        if frame is None:
            errors.append(f"Không đọc được ảnh: {image_path.name}")
            continue

        label_path = label_directory / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue  # đã ghi nhận ở mục 3

        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        # --- 12: file nhãn rỗng là HỢP LỆ (frame thật sự không có xe) ---
        if not lines:
            empty_label_files += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            where = f"{label_path.name}:{line_number}"
            values = line.split()

            # --- 6: đúng 5 trường ---
            if len(values) != 5:
                errors.append(
                    f"{where}: có {len(values)} trường, cần đúng 5 "
                    f"(class_id x_center y_center width height)"
                )
                continue

            # --- 7: class_id hợp lệ ---
            try:
                class_id = int(values[0])
            except ValueError:
                errors.append(f"{where}: class_id không phải số nguyên: {values[0]!r}")
                continue
            if class_id not in allowed_class_ids:
                errors.append(
                    f"{where}: class_id={class_id} không hợp lệ, chỉ chấp nhận "
                    f"{sorted(allowed_class_ids)} (xem evaluation/classes.txt)"
                )
                continue

            # --- 8: bốn toạ độ phải là số hữu hạn ---
            try:
                cx, cy, bw, bh = (float(v) for v in values[1:5])
            except ValueError:
                errors.append(f"{where}: toạ độ không phải số: {values[1:5]}")
                continue
            if not all(map(_is_finite, (cx, cy, bw, bh))):
                errors.append(f"{where}: toạ độ chứa NaN hoặc vô cực")
                continue

            # --- 9: toạ độ chuẩn hoá phải nằm trong [0, 1] ---
            out_of_range = [
                name for name, value in
                (("x_center", cx), ("y_center", cy), ("width", bw), ("height", bh))
                if not 0.0 <= value <= 1.0
            ]
            if out_of_range:
                errors.append(
                    f"{where}: {', '.join(out_of_range)} nằm ngoài [0, 1] "
                    f"— nhãn YOLO phải được chuẩn hoá theo kích thước ảnh"
                )
                continue

            # --- 10: width và height phải dương ---
            if bw <= 0 or bh <= 0:
                errors.append(f"{where}: width={bw} hoặc height={bh} không dương")
                continue

            # --- 11: box sau khi đổi sang xyxy không được suy biến ---
            image_height, image_width = frame.shape[:2]
            x1 = (cx - bw / 2) * image_width
            y1 = (cy - bh / 2) * image_height
            x2 = (cx + bw / 2) * image_width
            y2 = (cy + bh / 2) * image_height
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                errors.append(
                    f"{where}: box suy biến sau khi đổi sang pixel "
                    f"({x2 - x1:.2f}×{y2 - y1:.2f} px, cần >= 1×1)"
                )
                continue

            total_boxes += 1
            class_counter[class_id] += 1

    if errors:
        preview = "\n  - ".join(errors[:20])
        more = f"\n  ... và {len(errors) - 20} lỗi khác" if len(errors) > 20 else ""
        raise ValueError(
            f"Dataset KHÔNG hợp lệ ({len(errors)} lỗi):\n  - {preview}{more}\n\n"
            f"Xem evaluation/README.md để biết quy tắc gán nhãn. "
            f"Đánh giá đã dừng, KHÔNG sinh file CSV nào."
        )

    return {
        "number_of_images": len(image_paths),
        "number_of_label_files": len(label_paths),
        "number_of_ground_truth_boxes": total_boxes,
        "number_of_empty_label_files": empty_label_files,
        "boxes_per_class": {
            CLASS_NAMES.get(cid, str(cid)): n for cid, n in class_counter.items()
        },
    }


def _is_finite(value):
    """True nếu value là số hữu hạn (không phải NaN hay vô cực)."""
    return value == value and value not in (float("inf"), float("-inf"))


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


def _get_git_commit():
    """Lấy commit hash hiện tại, trả về None nếu không phải repo Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _get_package_version(module_name):
    """Lấy __version__ của một package, None nếu chưa cài."""
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except ImportError:
        return None


def _sha256_of_file(path):
    """Băm SHA256 của file, dùng để xác định chính xác trọng số model đã dùng.

    Đọc theo khối 1 MB thay vì nạp cả file vào RAM, vì file .pt có thể lớn.
    """
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_metadata(dataset_stats, model_path, conf, iou_match, device):
    """Gom toàn bộ thông tin cần để TÁI LẬP đúng kết quả này về sau.

    Không ghi cứng phiên bản thư viện: đọc trực tiếp từ môi trường đang chạy.
    Nếu sau này Ultralytics đổi mặc định (ví dụ ngưỡng NMS), file metadata này
    là bằng chứng duy nhất cho biết kết quả cũ được sinh ra dưới cấu hình nào.
    """
    return {
        "git_commit": _get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": "class-agnostic",
        "number_of_images": dataset_stats["number_of_images"],
        "number_of_ground_truth_boxes":
            dataset_stats["number_of_ground_truth_boxes"],
        "boxes_per_class": dataset_stats["boxes_per_class"],
        "number_of_empty_label_files":
            dataset_stats["number_of_empty_label_files"],
        "model_path": str(model_path),
        "model_sha256": _sha256_of_file(model_path),
        "yolo_conf_threshold": conf,
        "yolo_nms_iou_threshold": config.IOU_THRESHOLD,
        "matching_iou_threshold": iou_match,
        "max_det": config.MAX_DET,
        "classical_config": {
            "median_kernel_size": config.MEDIAN_KERNEL_SIZE,
            "min_contour_area": config.MIN_CONTOUR_AREA,
            "max_contour_area": config.MAX_CONTOUR_AREA,
            "max_aspect_ratio": config.MAX_ASPECT_RATIO,
            "sobel_edge_threshold": config.SOBEL_EDGE_THRESHOLD,
        },
        "device": device,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "numpy_version": _get_package_version("numpy"),
        "ultralytics_version": _get_package_version("ultralytics"),
        "torch_version": _get_package_version("torch"),
    }


def write_json(output_path, data):
    """Ghi dict ra file JSON, giữ nguyên tiếng Việt có dấu."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


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
    counter = {"total": 0}

    def check(name, actual, expected, tolerance=1e-6):
        counter["total"] += 1
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

    total = counter["total"]
    print()
    if failures:
        print(f"KẾT QUẢ: THẤT BẠI {len(failures)}/{total} kiểm thử: {failures}")
        return 1

    print(f"KẾT QUẢ: {total}/{total} KIỂM THỬ ĐỀU PASS")
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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Chỉ kiểm tra tính hợp lệ của dataset, không nạp model YOLO.",
    )
    args = parser.parse_args()

    if args.selftest:
        return run_selftest()

    image_directory = Path(args.images)
    label_directory = Path(args.labels)
    output_directory = Path(args.output)

    # BƯỚC 1 - Kiểm tra dataset TRƯỚC khi làm bất cứ việc gì khác.
    # Đặt trước cả việc nạp model vì nạp YOLO mất vài giây; nếu dataset sai
    # thì không có lý do gì bắt người dùng chờ.
    print("Đang kiểm tra dataset...")
    dataset_stats = validate_evaluation_dataset(
        image_directory, label_directory, ALLOWED_CLASS_IDS
    )
    print(f"  Số ảnh                  : {dataset_stats['number_of_images']}")
    print(f"  Số file nhãn            : {dataset_stats['number_of_label_files']}")
    print(f"  Tổng ground-truth box   : {dataset_stats['number_of_ground_truth_boxes']}")
    print(f"  File nhãn rỗng          : {dataset_stats['number_of_empty_label_files']}")
    print(f"  Phân bố theo lớp        : {dataset_stats['boxes_per_class']}")
    print("  Dataset HỢP LỆ.")
    print()

    if args.validate_only:
        print("Chế độ --validate-only: đã kiểm tra xong, không chạy đánh giá.")
        return 0

    image_paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

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
    write_json(
        output_directory / "run_metadata.json",
        build_run_metadata(
            dataset_stats=dataset_stats,
            model_path=args.model,
            conf=args.conf,
            iou_match=args.iou_match,
            device=yolo_detector.device,
        ),
    )

    # Kiểm tra bất biến: nếu một trong các đẳng thức này sai thì logic ghép
    # box đã hỏng, và mọi con số phía sau đều không đáng tin.
    for summary in (classical_summary, yolo_summary):
        assert summary["tp"] + summary["fn"] == \
            dataset_stats["number_of_ground_truth_boxes"], \
            f"{summary['method']}: TP + FN != tổng ground truth"
        assert 0.0 <= summary["precision"] <= 1.0, "Precision ngoài [0, 1]"
        assert 0.0 <= summary["recall"] <= 1.0, "Recall ngoài [0, 1]"
        assert 0.0 <= summary["f1"] <= 1.0, "F1 ngoài [0, 1]"

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
    # Bắt lỗi dữ liệu ở đây để in thông báo gọn gàng thay vì traceback dài.
    # Traceback chỉ hữu ích khi lỗi nằm trong code; còn lỗi thiếu nhãn hay sai
    # định dạng là lỗi dữ liệu của người dùng, họ cần đọc được ngay vấn đề là gì.
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print()
        print("LỖI:", error, file=sys.stderr)
        sys.exit(1)
