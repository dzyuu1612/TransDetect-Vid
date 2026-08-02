"""
tools/visualize_ground_truth.py
---------------------------------
Vẽ nhãn YOLO lên ảnh để con người KIỂM TRA BẰNG MẮT.

Vì sao cần công cụ này: `evaluate_yolo.py --validate-only` chỉ kiểm tra ĐỊNH
DẠNG — đủ file, class_id trong 0..3, toạ độ trong [0,1]. Nó hoàn toàn không
biết box có được vẽ đúng lên chiếc xe hay không. Một file nhãn lệch toàn bộ 50
pixel, hoặc gán nhầm bus thành truck, vẫn qua validator một cách sạch sẽ. Chỉ
có mắt người nhìn vào ảnh mới phát hiện được những lỗi đó.

Script chỉ ĐỌC ảnh gốc và ghi ảnh preview sang thư mục riêng, không bao giờ sửa
ảnh trong `evaluation/images/`.

Cách chạy:
    # xem nhãn sơ bộ trước khi sửa
    python tools/visualize_ground_truth.py --labels evaluation/prelabels

    # xem ground truth sau khi đã sửa tay
    python tools/visualize_ground_truth.py --labels evaluation/labels

    # xem đúng vài ảnh chỉ định
    python tools/visualize_ground_truth.py --frames frame_000000 frame_000306
"""

import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}

# Màu BGR riêng cho từng lớp để nhìn một cái là biết ngay có bị gán nhầm lớp
# hay không, không cần đọc chữ.
CLASS_COLORS = {
    0: (0, 255, 0),      # car       - xanh lá
    1: (0, 165, 255),    # motorcycle- cam
    2: (255, 0, 255),    # bus       - hồng
    3: (255, 0, 0),      # truck     - xanh dương
}


def read_yolo_label(label_path, image_width, image_height):
    """Đọc file nhãn YOLO và đổi sang box pixel xyxy.

    Mỗi dòng có dạng `class_id x_center y_center width height` với bốn toạ độ
    đã chuẩn hoá về [0, 1]. Trả về list `(class_id, x1, y1, x2, y2)`.
    """
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.strip().split()
        if len(values) < 5:
            continue

        class_id = int(values[0])
        center_x = float(values[1]) * image_width
        center_y = float(values[2]) * image_height
        box_width = float(values[3]) * image_width
        box_height = float(values[4]) * image_height

        boxes.append((
            class_id,
            int(center_x - box_width / 2),
            int(center_y - box_height / 2),
            int(center_x + box_width / 2),
            int(center_y + box_height / 2),
        ))

    return boxes


def draw_labels_on_image(frame, boxes):
    """Vẽ box, số thứ tự và tên lớp lên một bản sao của khung hình.

    Số thứ tự giúp đối chiếu nhanh giữa ảnh preview và từng dòng trong file
    nhãn: thấy box số 7 sai thì sửa đúng dòng thứ 8 của file .txt.
    """
    output = frame.copy()

    for index, (class_id, x1, y1, x2, y2) in enumerate(boxes):
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        name = CLASS_NAMES.get(class_id, f"id{class_id}")

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{index}:{name}",
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    return output


def count_boxes(label_path):
    """Đếm số dòng nhãn hợp lệ trong một file, dùng để tìm ảnh đông xe nhất."""
    if not label_path.exists():
        return 0

    return sum(
        1 for line in label_path.read_text(encoding="utf-8").splitlines()
        if len(line.strip().split()) >= 5
    )


def choose_frames_to_preview(image_paths, labels_dir, sample_count, seed):
    """Chọn ảnh để xem: một phần ngẫu nhiên cố định seed + các ảnh đông xe nhất.

    Vì sao trộn hai nhóm: ảnh ngẫu nhiên cho thấy chất lượng nhãn nói chung,
    còn ảnh đông xe nhất là nơi dễ sai nhất (box chồng nhau, xe bị che khuất,
    xe nhỏ ở xa). Chỉ xem ảnh ngẫu nhiên thì rất dễ bỏ lọt đúng những frame khó.

    Seed cố định để hai người cùng chạy sẽ xem đúng cùng một tập ảnh.
    """
    random_generator = random.Random(seed)
    random_sample = random_generator.sample(
        image_paths, min(sample_count, len(image_paths))
    )

    busiest = sorted(
        image_paths,
        key=lambda path: count_boxes(labels_dir / f"{path.stem}.txt"),
        reverse=True,
    )[:sample_count]

    # dict.fromkeys vừa loại trùng vừa giữ nguyên thứ tự xuất hiện.
    return list(dict.fromkeys(random_sample + busiest))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Vẽ nhãn YOLO lên ảnh để kiểm tra bằng mắt. Validator chỉ kiểm tra "
            "định dạng, không biết box có đúng chiếc xe hay không."
        )
    )
    parser.add_argument("--images", default="evaluation/images",
                        help="Thư mục ảnh gốc (chỉ đọc, không sửa).")
    parser.add_argument("--labels", default="evaluation/labels",
                        help="Thư mục nhãn cần xem.")
    parser.add_argument("--output", default="evaluation/label_previews",
                        help="Thư mục ghi ảnh preview.")
    parser.add_argument("--sample", type=int, default=10,
                        help="Số ảnh ngẫu nhiên và số ảnh đông xe nhất.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed cố định để chọn lại đúng cùng tập ảnh.")
    parser.add_argument("--frames", nargs="*", default=None,
                        help="Chỉ vẽ đúng các frame này, ví dụ frame_000000.")
    parser.add_argument("--all", action="store_true",
                        help="Vẽ toàn bộ ảnh thay vì chỉ một mẫu.")
    return parser


def main():
    arguments = build_parser().parse_args()

    images_dir = Path(arguments.images)
    labels_dir = Path(arguments.labels)
    output_dir = Path(arguments.output)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục nhãn: {labels_dir}")

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"Không có ảnh nào trong {images_dir}")

    if arguments.frames:
        wanted = set(arguments.frames)
        selected = [path for path in image_paths if path.stem in wanted]
        missing = wanted - {path.stem for path in selected}
        if missing:
            raise ValueError(f"Không tìm thấy frame: {sorted(missing)}")
    elif arguments.all:
        selected = image_paths
    else:
        selected = choose_frames_to_preview(
            image_paths, labels_dir, arguments.sample, arguments.seed
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    frames_without_label = []

    for image_path in selected:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")

        image_height, image_width = frame.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            frames_without_label.append(image_path.stem)

        boxes = read_yolo_label(label_path, image_width, image_height)
        total_boxes += len(boxes)

        preview = draw_labels_on_image(frame, boxes)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_preview.jpg"), preview)

        print(f"  {image_path.stem}: {len(boxes)} box")

    print()
    print("KẾT QUẢ")
    print("=" * 60)
    print(f"  Số ảnh preview : {len(selected)}")
    print(f"  Tổng box vẽ    : {total_boxes}")
    print(f"  Thư mục        : {output_dir}")
    print(f"  Màu: car=xanh lá, motorcycle=cam, bus=hồng, truck=xanh dương")

    if frames_without_label:
        print()
        print(f"  CẢNH BÁO: {len(frames_without_label)} ảnh KHÔNG có file nhãn: "
              f"{frames_without_label[:5]}")

    print()
    print("Mở các ảnh trên và kiểm tra: box có ôm đúng xe không, có xe nào bị")
    print("bỏ sót không, màu box có đúng loại xe không.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as error:
        print()
        print("LỖI:", error, file=sys.stderr)
        sys.exit(1)
