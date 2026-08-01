"""
evaluate_yolo.py
------------------
Đánh giá định lượng RIÊNG cho nhánh YOLO11 trên tập frame có ground truth.

Vì sao chỉ chấm YOLO11 mà không chấm nhánh truyền thống: YOLO11 sinh bounding
box kèm NHÃN LỚP và CONFIDENCE, nên tính được Precision/Recall/F1 có phân biệt
lớp và dựng được đường Precision-Recall để tính AP/mAP. Nhánh truyền thống chỉ
sinh vùng ứng viên từ threshold/Sobel/contour: không có lớp và không có
confidence chuẩn, nên không thể xếp hạng prediction để dựng đường PR. Ép nó có
mAP bằng một confidence giả (diện tích contour, hằng số 1,0...) sẽ ra con số
chạy được nhưng sai ý nghĩa. Nhánh truyền thống vẫn giữ trong repo để minh hoạ
chuỗi phép biến đổi và đo FPS — xem `evaluate_pipelines.py` cho phép so sánh
ĐỊNH VỊ class-agnostic tuỳ chọn.

Chỉ số báo cáo:
    - Precision / Recall / F1 tại đúng cấu hình demo (conf=0,25; IoU khớp 0,50).
    - AP@0.50 và AP@0.50:0.95 cho từng lớp.
    - mAP@0.50 và mAP@0.50:0.95 (chỉ số chính).

Script này CHỈ ĐỌC `Yolo11VehicleDetector` hiện có, không sửa thuật toán phát
hiện. Không dùng pycocotools/scikit-learn và không gọi API nội bộ có dấu `_`
của Ultralytics.

Cách chạy:
    python evaluate_yolo.py --images evaluation/images \
        --labels evaluation/labels --model yolo11n.pt --imgsz 640 \
        --conf 0.25 --predict-conf 0.001 --nms-iou 0.45 --match-iou 0.50 \
        --output evaluation/results_yolo

    python evaluate_yolo.py --selftest        # không cần ảnh, nhãn hay model
    python evaluate_yolo.py --validate-only   # chỉ kiểm tra dataset
"""

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

from src.transdetect import config


# Console Windows mặc định dùng cp1252, không mã hoá được tiếng Việt có dấu nên
# một lệnh print bình thường cũng có thể làm script dừng bằng UnicodeEncodeError
# sau khi đã chạy xong phần tính toán. Ép stdout/stderr về UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Bốn lớp đánh giá, khớp đúng thứ tự trong evaluation/classes.txt.
CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
EVALUATION_CLASSES = ["car", "motorcycle", "bus", "truck"]

# Ánh xạ tên lớp của model về bốn lớp đánh giá.
#
# ĐÂY LÀ ĐIỂM DỄ SAI NHẤT: `yolo11n.pt` được pre-train trên COCO nên class ID
# nội bộ của nó (car=2, motorcycle=3, bus=5, truck=7) KHÁC hoàn toàn với
# class ID của tập nhãn (car=0, motorcycle=1, bus=2, truck=3). Nếu đem hai bộ
# ID này so trực tiếp với nhau thì mọi prediction đều bị gán nhầm lớp — script
# vẫn chạy ra số, nhưng con số đó vô nghĩa. Vì vậy toàn bộ việc ghép lớp ở đây
# đi qua TÊN lớp đã chuẩn hoá, không bao giờ đi qua ID.
CLASS_ALIASES = {
    "car": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "container truck": "truck",
}

# Mười ngưỡng IoU của chỉ số mAP@0.50:0.95 (0,50; 0,55; ...; 0,95).
AP_IOU_THRESHOLDS = [round(0.50 + 0.05 * step, 2) for step in range(10)]

# Tên trọng số Ultralytics hợp lệ mà thư viện tự tải nếu máy chưa có file.
ULTRALYTICS_WEIGHT_PATTERN = re.compile(r"^yolo(v\d+|11|12)[nsmlx](-\w+)?\.pt$")


# ---------------------------------------------------------------------------
# Hình học
# ---------------------------------------------------------------------------

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


def normalize_class_name(class_name):
    """Đổi tên lớp của model về một trong bốn lớp đánh giá.

    Trả về None nếu tên không thuộc bốn lớp (ví dụ `person`, `bicycle`) — khi
    đó prediction bị bỏ qua chứ không bị tính thành False Positive, vì tập nhãn
    không hề gán những lớp đó nên phạt model là không công bằng.
    """
    if class_name is None:
        return None

    return CLASS_ALIASES.get(str(class_name).strip().lower())


# ---------------------------------------------------------------------------
# Đọc và kiểm tra dữ liệu
# ---------------------------------------------------------------------------

def _is_finite(value):
    """True nếu value là số hữu hạn (không phải NaN hay vô cực)."""
    return value == value and value not in (float("inf"), float("-inf"))


def load_ground_truth(label_path, image_width, image_height):
    """Đọc nhãn YOLO và đổi toạ độ chuẩn hoá sang xyxy pixel.

    Định dạng mỗi dòng: class_id x_center y_center width height, trong đó bốn
    giá trị toạ độ đã chuẩn hoá về [0, 1] theo kích thước ảnh.

    Khác với `evaluate_pipelines.py` (bỏ class_id vì chấm class-agnostic), ở đây
    class_id được GIỮ LẠI và đổi sang tên lớp: phép đánh giá này là class-aware,
    một prediction chỉ được tính đúng khi trùng cả vị trí lẫn lớp.

    Trả về list các dict `{"class_name": str, "bbox": [x1, y1, x2, y2]}`.
    """
    ground_truth_boxes = []

    if not label_path.exists():
        return ground_truth_boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.strip().split()
        if len(values) < 5:
            continue

        class_id = int(values[0])
        center_x = float(values[1]) * image_width
        center_y = float(values[2]) * image_height
        box_width = float(values[3]) * image_width
        box_height = float(values[4]) * image_height

        ground_truth_boxes.append({
            "class_name": CLASS_NAMES[class_id],
            "bbox": [
                center_x - box_width / 2,
                center_y - box_height / 2,
                center_x + box_width / 2,
                center_y + box_height / 2,
            ],
        })

    return ground_truth_boxes


def validate_dataset(images_dir, labels_dir):
    """Kiểm tra nghiêm ngặt tập ảnh + nhãn TRƯỚC khi chấm điểm.

    Vì sao phải dừng hẳn thay vì cảnh báo rồi chạy tiếp: một ảnh thiếu file nhãn
    sẽ bị hiểu là "frame này không có phương tiện nào", nên mọi box YOLO phát
    hiện trên ảnh đó đều thành False Positive. Chỉ vài ảnh thiếu nhãn là
    Precision đã sai lệch đáng kể mà không để lại dấu vết nào trong file CSV.
    Sai lệch âm thầm kiểu này nguy hiểm hơn nhiều so với việc script báo lỗi.

    Trả về dict thống kê. Ném ValueError kèm danh sách lỗi cụ thể nếu không hợp lệ.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    errors = []

    if not images_dir.is_dir():
        raise ValueError(f"Không tìm thấy thư mục ảnh: {images_dir}")
    if not labels_dir.is_dir():
        raise ValueError(f"Không tìm thấy thư mục nhãn: {labels_dir}")

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(
            f"Không có ảnh nào trong {images_dir}.\n"
            f"Chạy prepare_evaluation_frames.py để trích frame trước."
        )

    label_paths = sorted(labels_dir.glob("*.txt"))
    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}

    missing_labels = sorted(image_stems - label_stems)
    if missing_labels:
        errors.append(
            f"{len(missing_labels)} ảnh KHÔNG có file nhãn tương ứng: "
            f"{missing_labels[:5]}{' ...' if len(missing_labels) > 5 else ''}"
        )

    orphan_labels = sorted(label_stems - image_stems)
    if orphan_labels:
        errors.append(
            f"{len(orphan_labels)} file nhãn KHÔNG có ảnh tương ứng: "
            f"{orphan_labels[:5]}{' ...' if len(orphan_labels) > 5 else ''}"
        )

    total_boxes = 0
    empty_label_files = 0
    boxes_per_class = {name: 0 for name in EVALUATION_CLASSES}

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            errors.append(f"Không đọc được ảnh: {image_path.name}")
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue  # đã ghi nhận ở trên

        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        # File nhãn RỖNG là hợp lệ: frame thật sự không có phương tiện nào.
        if not lines:
            empty_label_files += 1
            continue

        image_height, image_width = frame.shape[:2]

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            where = f"{label_path.name}:{line_number}"
            values = line.split()

            if len(values) != 5:
                errors.append(
                    f"{where}: có {len(values)} trường, cần đúng 5 "
                    f"(class_id x_center y_center width height)"
                )
                continue

            try:
                class_id = int(values[0])
            except ValueError:
                errors.append(
                    f"{where}: class_id không phải số nguyên: {values[0]!r}"
                )
                continue
            if class_id not in CLASS_NAMES:
                errors.append(
                    f"{where}: class_id={class_id} không hợp lệ, chỉ chấp nhận "
                    f"{sorted(CLASS_NAMES)} (xem evaluation/classes.txt)"
                )
                continue

            try:
                cx, cy, bw, bh = (float(value) for value in values[1:5])
            except ValueError:
                errors.append(f"{where}: toạ độ không phải số: {values[1:5]}")
                continue
            if not all(map(_is_finite, (cx, cy, bw, bh))):
                errors.append(f"{where}: toạ độ chứa NaN hoặc vô cực")
                continue

            out_of_range = [
                name for name, value in
                (("x_center", cx), ("y_center", cy),
                 ("width", bw), ("height", bh))
                if not 0.0 <= value <= 1.0
            ]
            if out_of_range:
                errors.append(
                    f"{where}: {', '.join(out_of_range)} nằm ngoài [0, 1] "
                    f"— nhãn YOLO phải được chuẩn hoá theo kích thước ảnh"
                )
                continue

            if bw <= 0 or bh <= 0:
                errors.append(
                    f"{where}: width={bw} hoặc height={bh} không dương"
                )
                continue

            box_width_pixels = bw * image_width
            box_height_pixels = bh * image_height
            if box_width_pixels < 1.0 or box_height_pixels < 1.0:
                errors.append(
                    f"{where}: box suy biến sau khi đổi sang pixel "
                    f"({box_width_pixels:.2f}×{box_height_pixels:.2f} px, "
                    f"cần >= 1×1)"
                )
                continue

            total_boxes += 1
            boxes_per_class[CLASS_NAMES[class_id]] += 1

    if errors:
        preview = "\n  - ".join(errors[:20])
        more = f"\n  ... và {len(errors) - 20} lỗi khác" if len(errors) > 20 else ""
        raise ValueError(
            f"Dataset KHÔNG hợp lệ ({len(errors)} lỗi):\n  - {preview}{more}\n\n"
            f"Xem evaluation/README.md để biết quy tắc gán nhãn. "
            f"Đánh giá đã dừng, KHÔNG sinh file kết quả nào."
        )

    if total_boxes == 0:
        raise ValueError(
            "Toàn bộ file nhãn đều rỗng: tổng ground-truth box = 0.\n"
            "Không thể tính Precision/Recall/mAP khi không có đối tượng nào."
        )

    return {
        "number_of_images": len(image_paths),
        "number_of_label_files": len(label_paths),
        "number_of_ground_truth_boxes": total_boxes,
        "number_of_empty_label_files": empty_label_files,
        "boxes_per_class": boxes_per_class,
    }


# ---------------------------------------------------------------------------
# Suy luận
# ---------------------------------------------------------------------------

def predict_all_images(detector, image_paths, labels_dir, low_confidence,
                       nms_iou, image_size, max_det):
    """Chạy YOLO11 MỘT LẦN trên toàn tập ở ngưỡng confidence rất thấp.

    Vì sao chạy ở `low_confidence` (0,001) thay vì 0,25: đường Precision-Recall
    dùng để tính AP cần cả những prediction điểm thấp ở phần đuôi. Nếu chỉ chạy
    ở 0,25 thì đường PR bị cắt cụt và AP thấp hơn giá trị thật. Chỉ số
    Precision/Recall/F1 vẫn được tính ở 0,25 bằng cách LỌC LẠI danh sách này,
    nên toàn bộ báo cáo chỉ tốn đúng một lượt suy luận.

    Trả về (predictions_by_image, ground_truths_by_image), cả hai đều là dict
    khoá theo tên ảnh.
    """
    predictions_by_image = {}
    ground_truths_by_image = {}

    for index, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")

        image_height, image_width = frame.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"

        ground_truths_by_image[image_path.name] = load_ground_truth(
            label_path, image_width, image_height
        )

        detections = detector.detect_frame(
            frame,
            conf_threshold=low_confidence,
            iou_threshold=nms_iou,
            max_det=max_det,
            img_size=image_size,
        )

        frame_predictions = []
        for detection in detections:
            class_name = normalize_class_name(detection["class"])
            if class_name is None:
                continue
            frame_predictions.append({
                "class_name": class_name,
                "confidence": float(detection["confidence"]),
                "bbox": [float(value) for value in detection["bbox"]],
            })

        predictions_by_image[image_path.name] = frame_predictions

        if index % 10 == 0 or index == len(image_paths):
            print(f"  Đã suy luận {index}/{len(image_paths)} ảnh")

    return predictions_by_image, ground_truths_by_image


# ---------------------------------------------------------------------------
# Precision / Recall / F1 tại ngưỡng vận hành
# ---------------------------------------------------------------------------

def match_at_threshold(frame_predictions, frame_ground_truths, iou_threshold,
                       conf_threshold):
    """Ghép một-một trong MỘT frame, có phân biệt lớp.

    Thuật toán tham lam: xét mọi cặp (prediction, ground truth) CÙNG LỚP, sắp
    theo IoU giảm dần, rồi lần lượt chốt cặp có IoU cao nhất còn khả dụng. Mỗi
    prediction và mỗi ground truth chỉ được dùng đúng một lần — nhờ vậy hai box
    chồng lên cùng một chiếc xe chỉ được tính một TP, box thừa thành FP.

    Điều kiện lọc `confidence >= conf_threshold` được áp dụng TRƯỚC khi ghép, vì
    đây là chỉ số ở ngưỡng vận hành: prediction mà demo không hiển thị thì cũng
    không được tính vào Precision/Recall.

    Trả về dict `{class_name: {"tp": ..., "fp": ..., "fn": ...}}`.
    """
    kept_predictions = [
        prediction for prediction in frame_predictions
        if prediction["confidence"] >= conf_threshold
    ]

    counts = {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in EVALUATION_CLASSES
    }

    for class_name in EVALUATION_CLASSES:
        predictions = [
            prediction for prediction in kept_predictions
            if prediction["class_name"] == class_name
        ]
        ground_truths = [
            ground_truth for ground_truth in frame_ground_truths
            if ground_truth["class_name"] == class_name
        ]

        candidate_pairs = []
        for prediction_index, prediction in enumerate(predictions):
            for ground_truth_index, ground_truth in enumerate(ground_truths):
                iou = calculate_iou(prediction["bbox"], ground_truth["bbox"])
                candidate_pairs.append(
                    (iou, prediction_index, ground_truth_index)
                )

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        matched_predictions = set()
        matched_ground_truths = set()
        true_positive = 0

        for iou, prediction_index, ground_truth_index in candidate_pairs:
            # Danh sách đã sắp giảm dần nên gặp cặp dưới ngưỡng là dừng được
            # ngay, mọi cặp phía sau chắc chắn cũng dưới ngưỡng.
            if iou < iou_threshold:
                break
            if prediction_index in matched_predictions:
                continue
            if ground_truth_index in matched_ground_truths:
                continue

            matched_predictions.add(prediction_index)
            matched_ground_truths.add(ground_truth_index)
            true_positive += 1

        counts[class_name]["tp"] = true_positive
        counts[class_name]["fp"] = len(predictions) - true_positive
        counts[class_name]["fn"] = len(ground_truths) - true_positive

    return counts


def calculate_precision_recall_f1(true_positive, false_positive, false_negative):
    """Tính Precision, Recall và F1 từ TP/FP/FN.

    Precision = TP / (TP + FP): trong các box phát hiện, bao nhiêu box đúng.
    Recall    = TP / (TP + FN): trong các xe thật, tìm được bao nhiêu.
    F1        = trung bình điều hoà của Precision và Recall.

    Mọi mẫu số bằng 0 đều trả về 0.0 thay vì lỗi chia cho 0.
    """
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return precision, recall, f1


# ---------------------------------------------------------------------------
# AP và mAP
# ---------------------------------------------------------------------------

def interpolate_average_precision(recalls, precisions):
    """Tính AP bằng nội suy 101 điểm recall (0,00; 0,01; ...; 1,00).

    Tại mỗi mức recall `r`, precision được lấy là GIÁ TRỊ LỚN NHẤT của mọi
    precision ứng với recall >= r (precision envelope). Cách này loại bỏ các
    răng cưa của đường PR thô và là phương pháp COCO đang dùng; nếu không có
    điểm nào đạt recall >= r thì precision tại đó bằng 0.

    AP là trung bình cộng của 101 giá trị precision đã nội suy.
    """
    recall_levels = [level / 100 for level in range(101)]
    total = 0.0

    for level in recall_levels:
        candidates = [
            precision
            for precision, recall in zip(precisions, recalls)
            if recall >= level
        ]
        total += max(candidates) if candidates else 0.0

    return total / len(recall_levels)


def calculate_ap_for_class(predictions_by_image, ground_truths_by_image,
                           class_name, iou_threshold):
    """Tính AP của MỘT lớp tại MỘT ngưỡng IoU.

    Prediction được xếp theo confidence GIẢM DẦN rồi ghép lần lượt với ground
    truth chưa được ghép của cùng ảnh, cùng lớp. Thứ tự theo confidence là bắt
    buộc: AP đo chất lượng XẾP HẠNG của model, nên một prediction sai có
    confidence cao đứng trước sẽ kéo precision xuống ở mọi mức recall phía sau.

    Trả về None nếu lớp này không có ground-truth box nào. KHÔNG trả về 0.0,
    vì 0.0 sẽ bị tính vào trung bình và kéo mAP xuống sai lệch — một lớp không
    xuất hiện trong tập đánh giá thì đơn giản là không đo được.
    """
    total_ground_truths = sum(
        1
        for ground_truths in ground_truths_by_image.values()
        for ground_truth in ground_truths
        if ground_truth["class_name"] == class_name
    )

    if total_ground_truths == 0:
        return None

    flat_predictions = []
    for image_name, predictions in predictions_by_image.items():
        for prediction in predictions:
            if prediction["class_name"] == class_name:
                flat_predictions.append((
                    prediction["confidence"],
                    image_name,
                    prediction["bbox"],
                ))

    flat_predictions.sort(key=lambda item: item[0], reverse=True)

    matched_ground_truths = {
        image_name: set() for image_name in ground_truths_by_image
    }

    recalls = []
    precisions = []
    cumulative_true_positive = 0
    cumulative_false_positive = 0

    for _, image_name, predicted_box in flat_predictions:
        frame_ground_truths = ground_truths_by_image.get(image_name, [])

        best_iou = 0.0
        best_index = -1
        for index, ground_truth in enumerate(frame_ground_truths):
            if ground_truth["class_name"] != class_name:
                continue
            if index in matched_ground_truths[image_name]:
                continue
            iou = calculate_iou(predicted_box, ground_truth["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index >= 0 and best_iou >= iou_threshold:
            matched_ground_truths[image_name].add(best_index)
            cumulative_true_positive += 1
        else:
            cumulative_false_positive += 1

        recalls.append(cumulative_true_positive / total_ground_truths)
        precisions.append(
            cumulative_true_positive
            / (cumulative_true_positive + cumulative_false_positive)
        )

    if not flat_predictions:
        return 0.0

    return interpolate_average_precision(recalls, precisions)


def calculate_map(predictions_by_image, ground_truths_by_image):
    """Tính AP theo lớp và mAP@0.50 / mAP@0.50:0.95 cho toàn tập.

    Chỉ những lớp CÓ ground truth mới tham gia phép lấy trung bình; lớp không
    có ground truth được ghi None để bảng kết quả hiển thị `N/A`.

    Trả về (ap_per_class, map50, map50_95) với `ap_per_class[class_name]` là
    dict `{"ap50": ..., "ap50_95": ...}`.
    """
    ap_per_class = {}

    for class_name in EVALUATION_CLASSES:
        ap_by_threshold = [
            calculate_ap_for_class(
                predictions_by_image,
                ground_truths_by_image,
                class_name,
                iou_threshold,
            )
            for iou_threshold in AP_IOU_THRESHOLDS
        ]

        if ap_by_threshold[0] is None:
            ap_per_class[class_name] = {"ap50": None, "ap50_95": None}
            continue

        ap_per_class[class_name] = {
            "ap50": ap_by_threshold[0],
            "ap50_95": sum(ap_by_threshold) / len(ap_by_threshold),
        }

    measured = [
        value for value in ap_per_class.values()
        if value["ap50"] is not None
    ]

    if not measured:
        return ap_per_class, None, None

    map50 = sum(value["ap50"] for value in measured) / len(measured)
    map50_95 = sum(value["ap50_95"] for value in measured) / len(measured)

    return ap_per_class, map50, map50_95


# ---------------------------------------------------------------------------
# Ghi file
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    """Ghi danh sách dictionary ra CSV.

    Dùng encoding utf-8-sig để Excel trên Windows mở đúng tiếng Việt.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    """Ghi dict ra file JSON, giữ nguyên tiếng Việt có dấu."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _format_metric(value):
    """Làm tròn 4 chữ số, đổi None thành chuỗi 'N/A' cho bảng CSV."""
    return "N/A" if value is None else round(value, 4)


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
    """Băm SHA256 của file, dùng để xác định chính xác trọng số đã dùng.

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


def build_run_metadata(arguments, dataset_stats, device, prediction_count):
    """Gom toàn bộ thông tin cần để TÁI LẬP đúng kết quả này về sau.

    Không ghi cứng phiên bản thư viện: đọc trực tiếp từ môi trường đang chạy.
    Nếu sau này Ultralytics đổi mặc định, file metadata này là bằng chứng duy
    nhất cho biết kết quả cũ được sinh ra dưới cấu hình nào.
    """
    return {
        "evaluation_scope": "YOLO11 class-aware object detection",
        "dataset_name": "TransDetect internal test set",
        "class_names": EVALUATION_CLASSES,
        "image_size": arguments.imgsz,
        "report_confidence_threshold": arguments.conf,
        "prediction_collection_threshold": arguments.predict_conf,
        "nms_iou_threshold": arguments.nms_iou,
        "matching_iou_threshold": arguments.match_iou,
        "ap_iou_thresholds": AP_IOU_THRESHOLDS,
        "max_det": arguments.max_det,
        "number_of_images": dataset_stats["number_of_images"],
        "number_of_ground_truth_boxes":
            dataset_stats["number_of_ground_truth_boxes"],
        "number_of_empty_label_files":
            dataset_stats["number_of_empty_label_files"],
        "boxes_per_class": dataset_stats["boxes_per_class"],
        "number_of_collected_predictions": prediction_count,
        "model_path": str(arguments.model),
        "model_sha256": _sha256_of_file(arguments.model),
        "device": device,
        "git_commit": _get_git_commit(),
        "command": " ".join(sys.argv),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "numpy_version": _get_package_version("numpy"),
        "torch_version": _get_package_version("torch"),
        "ultralytics_version": _get_package_version("ultralytics"),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_selftest():
    """Kiểm thử các hàm thuần tính toán: IoU, alias, matching, P/R/F1, AP.

    Không cần ảnh, nhãn hay model — chạy được ngay cả khi chưa có ground truth.
    """
    failures = []
    counter = {"total": 0}

    def check(name, actual, expected, tolerance=1e-6):
        counter["total"] += 1
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            ok = actual is not None and abs(actual - expected) <= tolerance
        else:
            ok = actual == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: "
              f"được {actual}, mong đợi {expected}")
        if not ok:
            failures.append(name)

    def box(class_name, coordinates, confidence=None):
        item = {"class_name": class_name, "bbox": coordinates}
        if confidence is not None:
            item["confidence"] = confidence
        return item

    print("1) calculate_iou")
    check("box trùng khít", calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
    check("box rời nhau", calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
    check("chồng một nửa",
          calculate_iou([0, 0, 100, 100], [50, 0, 150, 100]), 5000 / 15000)
    check("box suy biến không chia cho 0",
          calculate_iou([5, 5, 5, 5], [5, 5, 5, 5]), 0.0)
    check("ví dụ Mục 2.6.5",
          calculate_iou([0, 0, 10, 10], [2, 0, 14, 10]),
          80 / (100 + 120 - 80), tolerance=1e-4)

    print("2) normalize_class_name (alias theo TÊN, không theo COCO ID)")
    check("motorbike -> motorcycle",
          normalize_class_name("motorbike"), "motorcycle")
    check("container truck -> truck",
          normalize_class_name("container truck"), "truck")
    check("Car viết hoa -> car", normalize_class_name("Car"), "car")
    check("bus giữ nguyên", normalize_class_name("bus"), "bus")
    check("person không thuộc 4 lớp", normalize_class_name("person"), None)
    check("None an toàn", normalize_class_name(None), None)

    print("3) match_at_threshold — class-aware, một-một")
    ground_truths = [box("car", [0, 0, 10, 10]), box("car", [100, 100, 110, 110])]

    counts = match_at_threshold(
        [box("car", [0, 0, 10, 10], 0.9), box("car", [100, 100, 110, 110], 0.9)],
        ground_truths, 0.5, 0.25,
    )
    check("khớp hoàn hảo TP", counts["car"]["tp"], 2)
    check("khớp hoàn hảo FP", counts["car"]["fp"], 0)
    check("khớp hoàn hảo FN", counts["car"]["fn"], 0)

    counts = match_at_threshold([], ground_truths, 0.5, 0.25)
    check("không prediction -> toàn FN", counts["car"]["fn"], 2)
    check("không prediction -> TP", counts["car"]["tp"], 0)

    counts = match_at_threshold(
        [box("car", [0, 0, 10, 10], 0.9)], [], 0.5, 0.25
    )
    check("không ground truth -> toàn FP", counts["car"]["fp"], 1)

    # Hai prediction cùng chồng lên MỘT ground truth: chỉ 1 TP, 1 FP.
    counts = match_at_threshold(
        [box("car", [0, 0, 10, 10], 0.9), box("car", [1, 1, 11, 11], 0.8)],
        [box("car", [0, 0, 10, 10])], 0.5, 0.25,
    )
    check("hai pred một GT -> TP", counts["car"]["tp"], 1)
    check("hai pred một GT -> FP", counts["car"]["fp"], 1)
    check("hai pred một GT -> FN", counts["car"]["fn"], 0)

    # Trùng vị trí nhưng SAI LỚP: 1 FP ở lớp đoán, 1 FN ở lớp thật.
    counts = match_at_threshold(
        [box("truck", [0, 0, 10, 10], 0.9)],
        [box("car", [0, 0, 10, 10])], 0.5, 0.25,
    )
    check("sai lớp -> car TP", counts["car"]["tp"], 0)
    check("sai lớp -> car FN", counts["car"]["fn"], 1)
    check("sai lớp -> truck FP", counts["truck"]["fp"], 1)

    counts = match_at_threshold(
        [box("car", [0, 0, 100, 100], 0.9)],
        [box("car", [80, 0, 180, 100])], 0.5, 0.25,
    )
    check("dưới ngưỡng IoU -> TP", counts["car"]["tp"], 0)
    check("dưới ngưỡng IoU -> FP", counts["car"]["fp"], 1)
    check("dưới ngưỡng IoU -> FN", counts["car"]["fn"], 1)

    # Prediction dưới conf=0.25 không được tham gia P/R/F1.
    counts = match_at_threshold(
        [box("car", [0, 0, 10, 10], 0.10)],
        [box("car", [0, 0, 10, 10])], 0.5, 0.25,
    )
    check("dưới conf -> TP", counts["car"]["tp"], 0)
    check("dưới conf -> FP", counts["car"]["fp"], 0)
    check("dưới conf -> FN", counts["car"]["fn"], 1)

    print("4) calculate_precision_recall_f1")
    precision, recall, f1 = calculate_precision_recall_f1(80, 20, 40)
    check("Precision", precision, 0.80)
    check("Recall", recall, 80 / 120, tolerance=1e-4)
    check("F1", f1, 2 * 0.8 * (80 / 120) / (0.8 + 80 / 120), tolerance=1e-4)

    precision, recall, f1 = calculate_precision_recall_f1(0, 0, 10)
    check("toàn FN Precision", precision, 0.0)
    check("toàn FN Recall", recall, 0.0)
    check("toàn FN F1", f1, 0.0)

    print("5) AP và mAP")
    perfect_ground_truths = {"a.jpg": [box("car", [0, 0, 10, 10])]}
    perfect_predictions = {"a.jpg": [box("car", [0, 0, 10, 10], 0.9)]}
    check("AP dự đoán hoàn hảo = 1",
          calculate_ap_for_class(
              perfect_predictions, perfect_ground_truths, "car", 0.5), 1.0)

    # Một prediction SAI nhưng confidence CAO đứng trước prediction đúng phải
    # kéo AP xuống — đây là điểm mấu chốt phân biệt AP với Precision đơn thuần.
    noisy_predictions = {"a.jpg": [
        box("car", [500, 500, 510, 510], 0.99),
        box("car", [0, 0, 10, 10], 0.30),
    ]}
    noisy_ap = calculate_ap_for_class(
        noisy_predictions, perfect_ground_truths, "car", 0.5)
    check("AP giảm khi FP confidence cao đứng trước", noisy_ap < 1.0, True)
    check("AP vẫn dương vì cuối cùng vẫn tìm ra xe", noisy_ap > 0.0, True)

    # Lớp không có ground truth phải trả về None, KHÔNG phải 0.0.
    check("lớp không có GT -> None",
          calculate_ap_for_class(
              perfect_predictions, perfect_ground_truths, "bus", 0.5), None)

    ap_per_class, map50, map50_95 = calculate_map(
        perfect_predictions, perfect_ground_truths)
    check("mAP50 chỉ trung bình lớp có GT", map50, 1.0)
    check("mAP50-95 dự đoán hoàn hảo", map50_95, 1.0)
    check("bus vẫn là None trong bảng", ap_per_class["bus"]["ap50"], None)

    print("6) interpolate_average_precision")
    check("PR hoàn hảo -> AP = 1",
          interpolate_average_precision([1.0], [1.0]), 1.0)
    check("không recall nào -> AP = 0",
          interpolate_average_precision([0.0], [0.0]), 0.0)

    print("7) load_ground_truth giữ đúng lớp và đổi sang pixel")
    import tempfile
    with tempfile.TemporaryDirectory() as temporary_directory:
        label_path = Path(temporary_directory) / "frame.txt"
        # class_id=1 (motorcycle), tâm giữa ảnh, rộng/cao bằng nửa ảnh.
        label_path.write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        loaded = load_ground_truth(label_path, 200, 100)
        check("số box đọc được", len(loaded), 1)
        check("class_id=1 -> motorcycle", loaded[0]["class_name"], "motorcycle")
        check("x1 pixel", loaded[0]["bbox"][0], 50.0)
        check("y1 pixel", loaded[0]["bbox"][1], 25.0)
        check("x2 pixel", loaded[0]["bbox"][2], 150.0)
        check("y2 pixel", loaded[0]["bbox"][3], 75.0)

        check("file nhãn không tồn tại -> rỗng",
              len(load_ground_truth(
                  Path(temporary_directory) / "khong_co.txt", 200, 100)), 0)

    print("8) validate_dataset phải DỪNG khi nhãn sai")
    import numpy

    def make_dataset(label_text, directory):
        images_dir = Path(directory) / "images"
        labels_dir = Path(directory) / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        blank = numpy.zeros((100, 200, 3), dtype=numpy.uint8)
        cv2.imwrite(str(images_dir / "frame.jpg"), blank)
        if label_text is not None:
            (labels_dir / "frame.txt").write_text(label_text, encoding="utf-8")
        return images_dir, labels_dir

    def expect_rejected(name, label_text):
        with tempfile.TemporaryDirectory() as directory:
            images_dir, labels_dir = make_dataset(label_text, directory)
            try:
                validate_dataset(images_dir, labels_dir)
                check(name, "không báo lỗi", "ValueError")
            except ValueError:
                check(name, "ValueError", "ValueError")

    expect_rejected("thiếu file nhãn", None)
    expect_rejected("class_id ngoài 0..3", "9 0.5 0.5 0.2 0.2\n")
    expect_rejected("toạ độ ngoài [0,1]", "0 1.7 0.5 0.2 0.2\n")
    expect_rejected("thiếu trường", "0 0.5 0.5 0.2\n")
    expect_rejected("width không dương", "0 0.5 0.5 0.0 0.2\n")
    expect_rejected("toàn bộ nhãn rỗng", "")

    with tempfile.TemporaryDirectory() as directory:
        images_dir, labels_dir = make_dataset("2 0.5 0.5 0.4 0.4\n", directory)
        stats = validate_dataset(images_dir, labels_dir)
        check("nhãn hợp lệ -> số ảnh", stats["number_of_images"], 1)
        check("nhãn hợp lệ -> số box",
              stats["number_of_ground_truth_boxes"], 1)
        check("nhãn hợp lệ -> đếm đúng lớp bus",
              stats["boxes_per_class"]["bus"], 1)

    total = counter["total"]
    print()
    if failures:
        print(f"KẾT QUẢ: THẤT BẠI {len(failures)}/{total} kiểm thử: {failures}")
        return 1

    print(f"KẾT QUẢ: {total}/{total} KIỂM THỬ ĐỀU PASS")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Đánh giá định lượng YOLO11 (class-aware) bằng Precision/Recall/F1, "
            "AP@0.50 và mAP@0.50:0.95."
        )
    )
    parser.add_argument("--images", default="evaluation/images",
                        help="Thư mục ảnh đánh giá.")
    parser.add_argument("--labels", default="evaluation/labels",
                        help="Thư mục nhãn định dạng YOLO.")
    parser.add_argument("--model", default=config.DEFAULT_YOLO_MODEL,
                        help="Đường dẫn hoặc tên trọng số YOLO11.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kích thước model nhận sau letterbox.")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD,
                        help="Ngưỡng confidence dùng cho Precision/Recall/F1.")
    parser.add_argument("--predict-conf", type=float, default=0.001,
                        help="Ngưỡng thu prediction để dựng đường PR tính AP.")
    parser.add_argument("--nms-iou", type=float, default=config.IOU_THRESHOLD,
                        help="Ngưỡng IoU của NMS trong lúc suy luận.")
    parser.add_argument("--match-iou", type=float, default=0.50,
                        help="Ngưỡng IoU để coi prediction khớp ground truth.")
    parser.add_argument("--max-det", type=int, default=config.MAX_DET,
                        help="Số detection tối đa mỗi frame.")
    parser.add_argument("--output", default="evaluation/results_yolo",
                        help="Thư mục lưu kết quả.")
    parser.add_argument("--selftest", action="store_true",
                        help="Chạy kiểm thử các hàm tính toán rồi thoát.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Chỉ kiểm tra dataset, không nạp model YOLO.")
    return parser


def validate_arguments(arguments):
    """Kiểm tra tham số TRƯỚC khi nạp model hay đọc ảnh.

    Nạp YOLO mất vài giây; nếu tham số đã sai thì không có lý do gì bắt người
    dùng chờ rồi mới báo lỗi.
    """
    if not 0.0 <= arguments.conf <= 1.0:
        raise ValueError(f"--conf phải nằm trong [0, 1], đang nhận {arguments.conf}")

    if not 0.0 <= arguments.predict_conf <= arguments.conf:
        raise ValueError(
            f"--predict-conf phải nằm trong [0, --conf] = "
            f"[0, {arguments.conf}], đang nhận {arguments.predict_conf}.\n"
            f"Ngưỡng thu prediction phải THẤP HƠN ngưỡng báo cáo, nếu không "
            f"đường Precision-Recall sẽ bị cắt cụt và AP thấp hơn giá trị thật."
        )

    if not 0.0 < arguments.nms_iou <= 1.0:
        raise ValueError(
            f"--nms-iou phải nằm trong (0, 1], đang nhận {arguments.nms_iou}"
        )

    if not 0.0 < arguments.match_iou <= 1.0:
        raise ValueError(
            f"--match-iou phải nằm trong (0, 1], đang nhận {arguments.match_iou}"
        )

    if arguments.imgsz <= 0:
        raise ValueError(f"--imgsz phải dương, đang nhận {arguments.imgsz}")

    if arguments.max_det <= 0:
        raise ValueError(f"--max-det phải dương, đang nhận {arguments.max_det}")

    model_path = Path(arguments.model)
    if not model_path.is_file() and not ULTRALYTICS_WEIGHT_PATTERN.match(
        model_path.name
    ):
        raise ValueError(
            f"Không tìm thấy model {arguments.model} và tên này cũng không phải "
            f"trọng số Ultralytics hợp lệ (ví dụ yolo11n.pt) để tự tải về."
        )


def build_result_rows(arguments, dataset_stats, predictions_by_image,
                      ground_truths_by_image):
    """Tính toàn bộ chỉ số và dựng các hàng cho bốn file CSV.

    Trả về dict gồm `summary`, `per_class`, `per_frame`, `predictions`.
    """
    per_frame_rows = []
    totals_per_class = {
        name: {"tp": 0, "fp": 0, "fn": 0} for name in EVALUATION_CLASSES
    }

    for image_name in sorted(predictions_by_image):
        frame_predictions = predictions_by_image[image_name]
        frame_ground_truths = ground_truths_by_image[image_name]

        counts = match_at_threshold(
            frame_predictions,
            frame_ground_truths,
            arguments.match_iou,
            arguments.conf,
        )

        frame_true_positive = 0
        frame_false_positive = 0
        frame_false_negative = 0
        for class_name in EVALUATION_CLASSES:
            totals_per_class[class_name]["tp"] += counts[class_name]["tp"]
            totals_per_class[class_name]["fp"] += counts[class_name]["fp"]
            totals_per_class[class_name]["fn"] += counts[class_name]["fn"]
            frame_true_positive += counts[class_name]["tp"]
            frame_false_positive += counts[class_name]["fp"]
            frame_false_negative += counts[class_name]["fn"]

        per_frame_rows.append({
            "image_name": image_name,
            "ground_truth_boxes": len(frame_ground_truths),
            "predictions": sum(
                1 for prediction in frame_predictions
                if prediction["confidence"] >= arguments.conf
            ),
            "tp": frame_true_positive,
            "fp": frame_false_positive,
            "fn": frame_false_negative,
        })

    ap_per_class, map50, map50_95 = calculate_map(
        predictions_by_image, ground_truths_by_image
    )

    per_class_rows = []
    for class_name in EVALUATION_CLASSES:
        counts = totals_per_class[class_name]
        precision, recall, f1 = calculate_precision_recall_f1(
            counts["tp"], counts["fp"], counts["fn"]
        )
        per_class_rows.append({
            "class_name": class_name,
            "ground_truth_boxes":
                dataset_stats["boxes_per_class"][class_name],
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "ap50": _format_metric(ap_per_class[class_name]["ap50"]),
            "ap50_95": _format_metric(ap_per_class[class_name]["ap50_95"]),
        })

    total_true_positive = sum(row["tp"] for row in per_class_rows)
    total_false_positive = sum(row["fp"] for row in per_class_rows)
    total_false_negative = sum(row["fn"] for row in per_class_rows)
    precision, recall, f1 = calculate_precision_recall_f1(
        total_true_positive, total_false_positive, total_false_negative
    )

    summary_row = {
        "model": Path(arguments.model).name,
        "images": dataset_stats["number_of_images"],
        "ground_truth_boxes": dataset_stats["number_of_ground_truth_boxes"],
        "predictions": sum(row["predictions"] for row in per_frame_rows),
        "tp": total_true_positive,
        "fp": total_false_positive,
        "fn": total_false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map50": _format_metric(map50),
        "map50_95": _format_metric(map50_95),
    }

    prediction_rows = []
    for image_name in sorted(predictions_by_image):
        for prediction in predictions_by_image[image_name]:
            x1, y1, x2, y2 = prediction["bbox"]
            prediction_rows.append({
                "image_name": image_name,
                "class_name": prediction["class_name"],
                "confidence": round(prediction["confidence"], 6),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            })

    return {
        "summary": summary_row,
        "per_class": per_class_rows,
        "per_frame": per_frame_rows,
        "predictions": prediction_rows,
    }


def print_results(summary_row, per_class_rows, arguments):
    """In bảng kết quả gọn gàng ra terminal để chụp màn hình làm bằng chứng."""
    print()
    print("KẾT QUẢ ĐÁNH GIÁ YOLO11 (class-aware)")
    print("=" * 78)
    print(
        f"imgsz={arguments.imgsz} | conf báo cáo={arguments.conf} | "
        f"NMS IoU={arguments.nms_iou} | IoU khớp={arguments.match_iou}"
    )
    print("-" * 78)
    print(
        f"{'Lớp':<12}{'GT':>6}{'TP':>6}{'FP':>6}{'FN':>6}"
        f"{'P':>9}{'R':>9}{'F1':>9}{'AP50':>9}{'AP50-95':>10}"
    )
    print("-" * 78)

    for row in per_class_rows:
        print(
            f"{row['class_name']:<12}{row['ground_truth_boxes']:>6}"
            f"{row['tp']:>6}{row['fp']:>6}{row['fn']:>6}"
            f"{row['precision']:>9.4f}{row['recall']:>9.4f}{row['f1']:>9.4f}"
            f"{str(row['ap50']):>9}{str(row['ap50_95']):>10}"
        )

    print("-" * 78)
    print(
        f"{'TỔNG':<12}{summary_row['ground_truth_boxes']:>6}"
        f"{summary_row['tp']:>6}{summary_row['fp']:>6}{summary_row['fn']:>6}"
        f"{summary_row['precision']:>9.4f}{summary_row['recall']:>9.4f}"
        f"{summary_row['f1']:>9.4f}"
    )
    print("=" * 78)
    print(f"mAP@0.50       : {summary_row['map50']}")
    print(f"mAP@0.50:0.95  : {summary_row['map50_95']}   <- chỉ số chính")
    print("=" * 78)


def main():
    arguments = build_parser().parse_args()

    if arguments.selftest:
        return run_selftest()

    validate_arguments(arguments)

    images_dir = Path(arguments.images)
    labels_dir = Path(arguments.labels)
    output_dir = Path(arguments.output)

    # BƯỚC 1 - Kiểm tra dataset trước khi nạp model.
    print("Đang kiểm tra dataset...")
    dataset_stats = validate_dataset(images_dir, labels_dir)
    print(f"  Số ảnh                : {dataset_stats['number_of_images']}")
    print(f"  Số file nhãn          : {dataset_stats['number_of_label_files']}")
    print(f"  Tổng ground-truth box : "
          f"{dataset_stats['number_of_ground_truth_boxes']}")
    print(f"  File nhãn rỗng        : "
          f"{dataset_stats['number_of_empty_label_files']}")
    print(f"  Phân bố theo lớp      : {dataset_stats['boxes_per_class']}")
    print("  Dataset HỢP LỆ.")
    print()

    if arguments.validate_only:
        print("Chế độ --validate-only: đã kiểm tra xong, không chạy đánh giá.")
        return 0

    # BƯỚC 2 - Suy luận một lượt duy nhất ở ngưỡng confidence thấp.
    from src.transdetect.yolo_detector import Yolo11VehicleDetector

    print(f"Đang nạp model {arguments.model}...")
    detector = Yolo11VehicleDetector(arguments.model)
    print(f"  Thiết bị: {detector.device}")

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Đang chạy YOLO11 ở conf={arguments.predict_conf} "
          f"(thu prediction để tính AP)...")
    predictions_by_image, ground_truths_by_image = predict_all_images(
        detector=detector,
        image_paths=image_paths,
        labels_dir=labels_dir,
        low_confidence=arguments.predict_conf,
        nms_iou=arguments.nms_iou,
        image_size=arguments.imgsz,
        max_det=arguments.max_det,
    )

    # BƯỚC 3 - Tính chỉ số.
    print("Đang tính Precision/Recall/F1 và AP/mAP...")
    results = build_result_rows(
        arguments, dataset_stats, predictions_by_image, ground_truths_by_image
    )

    # Kiểm tra bất biến: nếu đẳng thức này sai thì logic ghép box đã hỏng và
    # mọi con số phía sau đều không đáng tin.
    assert (
        results["summary"]["tp"] + results["summary"]["fn"]
        == dataset_stats["number_of_ground_truth_boxes"]
    ), "TP + FN != tổng ground-truth box"
    for row in results["per_class"]:
        assert row["tp"] + row["fn"] == row["ground_truth_boxes"], (
            f"Lớp {row['class_name']}: TP + FN != số ground-truth box"
        )

    # BƯỚC 4 - Ghi file kết quả.
    write_csv(output_dir / "summary_metrics.csv", [results["summary"]])
    write_csv(output_dir / "per_class_metrics.csv", results["per_class"])
    write_csv(output_dir / "per_frame_metrics.csv", results["per_frame"])
    write_csv(output_dir / "predictions.csv", results["predictions"])
    write_json(
        output_dir / "run_metadata.json",
        build_run_metadata(
            arguments=arguments,
            dataset_stats=dataset_stats,
            device=detector.device,
            prediction_count=len(results["predictions"]),
        ),
    )

    print_results(results["summary"], results["per_class"], arguments)
    print(f"Đã lưu kết quả tại: {output_dir}")
    return 0


if __name__ == "__main__":
    # Bắt lỗi dữ liệu ở đây để in thông báo gọn gàng thay vì traceback dài.
    # Traceback chỉ hữu ích khi lỗi nằm trong code; lỗi thiếu nhãn hay sai tham
    # số là lỗi của người dùng, họ cần đọc được ngay vấn đề là gì.
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print()
        print("LỖI:", error, file=sys.stderr)
        sys.exit(1)
