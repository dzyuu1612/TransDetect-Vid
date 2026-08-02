"""
tools/prelabel_yolo.py
------------------------
Sinh nhãn SƠ BỘ (pre-label) bằng YOLO11 để rút ngắn công gán nhãn thủ công.

CẢNH BÁO QUAN TRỌNG
    File sinh ra ở đây KHÔNG PHẢI ground truth. Nếu lấy dự đoán của YOLO làm
    nhãn chuẩn rồi chấm điểm chính YOLO đó, kết quả sẽ là vòng lặp tự khẳng
    định: mọi chỉ số đều ra gần 1,0000 và không chứng minh được điều gì.

    Vì vậy script GHI VÀO `evaluation/prelabels/`, không bao giờ ghi thẳng vào
    `evaluation/labels/`. Chỉ sau khi con người mở từng ảnh, sửa box thiếu, box
    thừa, sai lớp và box không ôm sát, nhãn mới được chép sang
    `evaluation/labels/`.

Vì sao dùng confidence THẤP (0,10) thay vì 0,25 như lúc chạy demo:
    Ở bước gán nhãn, bỏ sót tốn công hơn nhiều so với thừa. Xoá một box sai chỉ
    mất một cú nhấp, còn phát hiện ra một chiếc xe mà model không vẽ box thì
    người gán nhãn phải tự tìm và tự vẽ. Hạ ngưỡng xuống 0,10 làm model gợi ý
    nhiều hơn, chấp nhận có box sai để đổi lấy ít việc vẽ tay hơn.

Script này chỉ dùng `Yolo11VehicleDetector` sẵn có, không viết detector thứ hai.

Cách chạy:
    python tools/prelabel_yolo.py --limit 3      # chạy thử vài ảnh trước
    python tools/prelabel_yolo.py                # chạy toàn bộ
    python tools/prelabel_yolo.py --overwrite    # ghi đè file đã có
"""

import argparse
import sys
from pathlib import Path

import cv2

# Cho phép chạy `python tools/prelabel_yolo.py` từ thư mục gốc của repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transdetect import config


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Bốn lớp đánh giá và ID tương ứng, khớp evaluation/classes.txt.
CLASS_NAME_TO_ID = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3}

# Đổi tên lớp của model về bốn lớp trên. Phải ánh xạ theo TÊN vì `yolo11n.pt`
# pre-train trên COCO dùng class ID nội bộ (car=2, motorcycle=3, bus=5,
# truck=7), khác hoàn toàn ID của tập nhãn (0, 1, 2, 3).
CLASS_ALIASES = {
    "car": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "container truck": "truck",
}


def normalize_class_name(class_name):
    """Đổi tên lớp của model về một trong bốn lớp đánh giá.

    Trả về None nếu tên không thuộc bốn lớp (ví dụ `person`, `bicycle`); khi đó
    prediction bị bỏ qua vì tập nhãn không gán những lớp đó.
    """
    if class_name is None:
        return None

    return CLASS_ALIASES.get(str(class_name).strip().lower())


def convert_box_to_yolo_line(class_id, bbox, image_width, image_height):
    """Đổi một box xyxy pixel sang một dòng nhãn YOLO đã chuẩn hoá.

    Định dạng trả về: `class_id x_center y_center width height`, đúng năm
    trường. KHÔNG ghi confidence vào file nhãn — file nhãn ground truth không
    có cột đó, thêm vào sẽ làm validator báo sai định dạng.

    Toạ độ được kẹp về [0, 1] vì box của YOLO có thể tràn ra ngoài mép ảnh vài
    pixel với xe đang đi vào/ra khỏi khung hình.
    """
    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2 / image_width
    center_y = (y1 + y2) / 2 / image_height
    box_width = (x2 - x1) / image_width
    box_height = (y2 - y1) / image_height

    center_x = min(max(center_x, 0.0), 1.0)
    center_y = min(max(center_y, 0.0), 1.0)
    box_width = min(max(box_width, 0.0), 1.0)
    box_height = min(max(box_height, 0.0), 1.0)

    return (
        f"{class_id} {center_x:.6f} {center_y:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )


def prelabel_one_image(detector, image_path, confidence, image_size, nms_iou,
                       max_det):
    """Chạy YOLO trên một ảnh và trả về danh sách dòng nhãn YOLO.

    Trả về (danh sách dòng, số box theo từng lớp).
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    image_height, image_width = frame.shape[:2]

    detections = detector.detect_frame(
        frame,
        conf_threshold=confidence,
        iou_threshold=nms_iou,
        max_det=max_det,
        img_size=image_size,
    )

    lines = []
    boxes_per_class = {name: 0 for name in CLASS_NAME_TO_ID}

    for detection in detections:
        class_name = normalize_class_name(detection["class"])
        if class_name is None:
            continue

        class_id = CLASS_NAME_TO_ID[class_name]
        lines.append(
            convert_box_to_yolo_line(
                class_id, detection["bbox"], image_width, image_height
            )
        )
        boxes_per_class[class_name] += 1

    return lines, boxes_per_class


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sinh nhãn sơ bộ bằng YOLO11 vào evaluation/prelabels/. "
            "ĐÂY KHÔNG PHẢI ground truth — người dùng phải kiểm tra thủ công."
        )
    )
    parser.add_argument("--images", default="evaluation/images",
                        help="Thư mục ảnh cần gán nhãn sơ bộ.")
    parser.add_argument("--output", default="evaluation/prelabels",
                        help="Thư mục ghi nhãn sơ bộ.")
    parser.add_argument("--model", default=config.DEFAULT_YOLO_MODEL,
                        help="Đường dẫn hoặc tên trọng số YOLO11.")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Ngưỡng confidence, để thấp cho đỡ bỏ sót.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kích thước model nhận sau letterbox.")
    parser.add_argument("--nms-iou", type=float, default=config.IOU_THRESHOLD,
                        help="Ngưỡng IoU của NMS.")
    parser.add_argument("--max-det", type=int, default=config.MAX_DET,
                        help="Số detection tối đa mỗi ảnh.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Chỉ xử lý N ảnh đầu, dùng để chạy thử.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Cho phép ghi đè file nhãn sơ bộ đã có.")
    return parser


def main():
    arguments = build_parser().parse_args()

    images_dir = Path(arguments.images)
    output_dir = Path(arguments.output)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"Không có ảnh nào trong {images_dir}")

    if arguments.limit is not None:
        image_paths = image_paths[:arguments.limit]

    # Chặn ghi đè ngoài ý muốn: nếu người dùng đã sửa tay một số file nhãn sơ bộ
    # thì chạy lại script sẽ xoá sạch công sức đó.
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        path for path in image_paths
        if (output_dir / f"{path.stem}.txt").exists()
    ]
    if existing and not arguments.overwrite:
        raise FileExistsError(
            f"{len(existing)} file nhãn sơ bộ đã tồn tại trong {output_dir}.\n"
            f"Nếu bạn đã sửa tay các file này, chạy lại sẽ ghi đè hết. "
            f"Thêm --overwrite nếu thực sự muốn sinh lại."
        )

    from src.transdetect.yolo_detector import Yolo11VehicleDetector

    print(f"Đang nạp model {arguments.model}...")
    detector = Yolo11VehicleDetector(arguments.model)
    print(f"  Thiết bị: {detector.device}")
    print(f"  conf={arguments.conf} | imgsz={arguments.imgsz} | "
          f"NMS IoU={arguments.nms_iou} | max_det={arguments.max_det}")
    print()

    total_boxes = 0
    total_per_class = {name: 0 for name in CLASS_NAME_TO_ID}

    for index, image_path in enumerate(image_paths, start=1):
        lines, boxes_per_class = prelabel_one_image(
            detector=detector,
            image_path=image_path,
            confidence=arguments.conf,
            image_size=arguments.imgsz,
            nms_iou=arguments.nms_iou,
            max_det=arguments.max_det,
        )

        label_path = output_dir / f"{image_path.stem}.txt"
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

        total_boxes += len(lines)
        for name, count in boxes_per_class.items():
            total_per_class[name] += count

        if index % 25 == 0 or index == len(image_paths):
            print(f"  Đã xử lý {index}/{len(image_paths)} ảnh")

    print()
    print("THỐNG KÊ NHÃN SƠ BỘ")
    print("=" * 60)
    print(f"  Số ảnh          : {len(image_paths)}")
    print(f"  Tổng box gợi ý  : {total_boxes}")
    if image_paths:
        print(f"  Trung bình      : {total_boxes / len(image_paths):.1f} box/ảnh")
    for name in CLASS_NAME_TO_ID:
        print(f"  {name:<15} : {total_per_class[name]}")
    print(f"  Thư mục         : {output_dir}")
    print()
    print("ĐÂY CHƯA PHẢI GROUND TRUTH")
    print("=" * 60)
    print("  Mở từng ảnh trong CVAT / LabelImg / Roboflow Annotate và sửa:")
    print("    - box thiếu (quan trọng nhất: xe nhỏ ở xa, xe đỗ, xe ở mép ảnh)")
    print("    - box thừa")
    print("    - sai lớp (hay gặp: bus bị gán thành truck)")
    print("    - box không ôm sát phương tiện")
    print("  Chỉ sau khi kiểm tra bằng mắt TOÀN BỘ ảnh mới chép nhãn sang")
    print("  evaluation/labels/ rồi chạy:")
    print("    python evaluate_yolo.py --validate-only")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print()
        print("LỖI:", error, file=sys.stderr)
        sys.exit(1)
