"""
tools/visualize_roi_ground_truth.py
-------------------------------------
Vẽ ground truth kèm ĐƯỜNG BIÊN ROI để làm bằng chứng cho phạm vi đánh giá.

Vì sao cần riêng công cụ này: báo cáo tuyên bố "chỉ đánh giá phương tiện có tâm
bounding box nằm trong 40% phía dưới khung hình". Đó là một tuyên bố cần chứng
minh được bằng hình ảnh — người chấm phải nhìn thấy đường biên nằm ở đâu, box
nào được tính và box nào bị loại. Một con số trong CSV không tự chứng minh được
điều đó.

Script CHỈ ĐỌC ảnh gốc và nhãn, ghi ảnh preview sang thư mục riêng. Không sửa
ảnh nguồn, không sửa nhãn, không thêm/xoá box nào.

Quy ước vẽ:
    - Đường ngang vàng tại y = roi_y_min * H, kèm chữ "Evaluation ROI".
    - Box TRONG ROI: vẽ đậm bằng màu của lớp.
    - Box NGOÀI ROI: vẽ mảnh màu xám, để thấy rõ chúng tồn tại nhưng không
      tham gia tính điểm. Không ẩn hẳn, vì ẩn đi sẽ khiến người xem tưởng tập
      nhãn vốn không có những box đó.

Cách chạy:
    python tools/visualize_roi_ground_truth.py --roi-y-min 0.60
"""

import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.visualize_ground_truth import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    read_yolo_label,
)


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


OUTSIDE_ROI_COLOR = (150, 150, 150)
ROI_LINE_COLOR = (0, 220, 220)


def is_box_inside_vertical_roi(box, image_height, roi_y_min):
    """Tâm box xyxy có nằm từ roi_y_min * H trở xuống hay không.

    Giữ đúng cùng quy tắc `>=` với hai evaluator: box có tâm ĐÚNG tại biên được
    tính là nằm trong ROI. Nếu preview và evaluator lệch nhau ở dấu so sánh thì
    hình minh hoạ sẽ không còn là bằng chứng cho con số.

    Hàm này chỉ dùng để QUYẾT ĐỊNH MÀU VẼ. Việc ĐẾM dùng
    `count_boxes_inside_roi_exact` trên toạ độ chuẩn hoá gốc — xem giải thích ở
    đó.
    """
    if roi_y_min is None:
        return True

    _, y1, _, y2 = box
    center_y = (y1 + y2) / 2
    return center_y >= roi_y_min * image_height


def read_normalized_centers(label_path):
    """Đọc (class_id, y_center chuẩn hoá) trực tiếp từ file nhãn.

    Vì sao không dùng lại box pixel đã làm tròn: `read_yolo_label` ép toạ độ về
    `int` để vẽ, nên tâm box bị dịch tới nửa pixel. Với box nằm sát biên ROI,
    nửa pixel đó đủ để đẩy box sang phía bên kia, khiến preview đếm ra số khác
    evaluator (thực tế đã lệch 8 box). Evaluator so sánh trên số thực, và vì
    tâm y theo pixel bằng đúng `y_center * H`, phép so sánh
    `center_y_pixel >= roi_y_min * H` tương đương chính xác với
    `y_center >= roi_y_min` trên giá trị chuẩn hoá gốc.
    """
    if not label_path.exists():
        return []

    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.strip().split()
        if len(values) < 5:
            continue
        rows.append((int(values[0]), float(values[2])))

    return rows


def draw_roi_overlay(frame, boxes, roi_y_min):
    """Vẽ đường biên ROI và toàn bộ box, phân biệt trong/ngoài ROI.

    Trả về (ảnh đã vẽ, số box trong ROI, số box ngoài ROI).
    """
    output = frame.copy()
    image_height, image_width = output.shape[:2]
    boundary_y = int(roi_y_min * image_height)

    inside_count = 0
    outside_count = 0

    for class_id, x1, y1, x2, y2 in boxes:
        if is_box_inside_vertical_roi((x1, y1, x2, y2), image_height, roi_y_min):
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output, CLASS_NAMES.get(class_id, str(class_id)),
                (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )
            inside_count += 1
        else:
            cv2.rectangle(output, (x1, y1), (x2, y2), OUTSIDE_ROI_COLOR, 1)
            outside_count += 1

    cv2.line(output, (0, boundary_y), (image_width, boundary_y),
             ROI_LINE_COLOR, 2)
    cv2.putText(
        output, f"Evaluation ROI: center_y/H >= {roi_y_min}  (y >= {boundary_y}px)",
        (10, boundary_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_LINE_COLOR, 2,
    )
    cv2.putText(
        output, f"trong ROI: {inside_count}   ngoai ROI (mo): {outside_count}",
        (10, boundary_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ROI_LINE_COLOR, 2,
    )

    return output, inside_count, outside_count


def count_boxes_inside_roi_exact(label_path, roi_y_min):
    """Đếm ground-truth box trong ROI, khớp CHÍNH XÁC với evaluator.

    Không cần đọc ảnh vì phép so sánh thực hiện trên toạ độ đã chuẩn hoá.
    """
    return sum(
        1 for _, y_center in read_normalized_centers(label_path)
        if y_center >= roi_y_min
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Vẽ ground truth kèm đường biên ROI làm bằng chứng phạm vi."
    )
    parser.add_argument("--images", default="evaluation/images")
    parser.add_argument("--labels", default="evaluation/labels")
    parser.add_argument("--output", default="evaluation/roi_previews")
    parser.add_argument("--roi-y-min", type=float, default=0.60,
                        help="Tỉ lệ chiều cao ảnh, lọc theo tâm box.")
    parser.add_argument("--sample", type=int, default=10,
                        help="Số ảnh seed cố định và số ảnh nhiều GT trong ROI.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    arguments = build_parser().parse_args()

    if not 0.0 <= arguments.roi_y_min <= 1.0:
        raise ValueError(
            f"--roi-y-min phải nằm trong [0, 1], đang nhận {arguments.roi_y_min}"
        )

    images_dir = Path(arguments.images)
    labels_dir = Path(arguments.labels)
    output_dir = Path(arguments.output)

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"Không có ảnh nào trong {images_dir}")

    # --- Thống kê trên TOÀN BỘ tập, không chỉ các ảnh được vẽ preview ---
    total_full_frame = 0
    total_inside = 0
    inside_per_class = {name: 0 for name in CLASS_NAMES.values()}

    for image_path in image_paths:
        rows = read_normalized_centers(labels_dir / f"{image_path.stem}.txt")
        total_full_frame += len(rows)
        for class_id, y_center in rows:
            if y_center >= arguments.roi_y_min:
                total_inside += 1
                inside_per_class[CLASS_NAMES.get(class_id, str(class_id))] += 1

    # --- Chọn ảnh preview: seed cố định + ảnh nhiều GT trong ROI nhất ---
    random_generator = random.Random(arguments.seed)
    random_sample = random_generator.sample(
        image_paths, min(arguments.sample, len(image_paths))
    )
    busiest = sorted(
        image_paths,
        key=lambda path: count_boxes_inside_roi_exact(
            labels_dir / f"{path.stem}.txt", arguments.roi_y_min
        ),
        reverse=True,
    )[:arguments.sample]
    selected = list(dict.fromkeys(random_sample + busiest))

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in selected:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        image_height, image_width = frame.shape[:2]
        boxes = read_yolo_label(
            labels_dir / f"{image_path.stem}.txt", image_width, image_height
        )
        preview, inside, outside = draw_roi_overlay(
            frame, boxes, arguments.roi_y_min
        )
        cv2.imwrite(str(output_dir / f"{image_path.stem}_roi.jpg"), preview)
        print(f"  {image_path.stem}: trong ROI {inside:3d} | ngoài ROI {outside:3d}")

    print()
    print("THỐNG KÊ GROUND TRUTH THEO ROI (toàn bộ tập)")
    print("=" * 60)
    print(f"  ROI                    : center_y/H >= {arguments.roi_y_min}")
    print(f"  Số ảnh                 : {len(image_paths)}")
    print(f"  Tổng GT full-frame     : {total_full_frame}")
    print(f"  Tổng GT TRONG ROI      : {total_inside}")
    print(f"  Tổng GT NGOÀI ROI      : {total_full_frame - total_inside}")
    if total_full_frame:
        print(f"  Tỉ lệ nằm trong ROI    : "
              f"{100 * total_inside / total_full_frame:.1f}%")
    print("  GT trong ROI theo lớp  :")
    for name, count in inside_per_class.items():
        print(f"    {name:<12}: {count}")
    print(f"  Thư mục preview        : {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as error:
        print()
        print("LỖI:", error, file=sys.stderr)
        sys.exit(1)
