"""
tools/prepare_cvat_dataset.py
-------------------------------
Đóng gói ảnh và nhãn sơ bộ thành một file ZIP để import vào CVAT.

Script này CHỈ đóng gói. Nó không chạy model, không sinh nhãn và tuyệt đối
không biến nhãn sơ bộ thành ground truth. Sau khi import, người gán nhãn vẫn
phải mở từng ảnh để sửa.

Cấu trúc ZIP tạo ra (định dạng Ultralytics YOLO Detection của CVAT):

    cvat_dataset.zip
    ├── data.yaml
    ├── train.txt
    ├── images/train/frame_000000.jpg
    └── labels/train/frame_000000.txt

Cách chạy:
    # đóng gói kèm nhãn sơ bộ để sửa cho nhanh
    python tools/prepare_cvat_dataset.py --labels evaluation/prelabels

    # đóng gói ảnh trắng, tự vẽ từ đầu
    python tools/prepare_cvat_dataset.py --labels ""
"""

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Thứ tự lớp phải khớp evaluation/classes.txt, nếu không CVAT sẽ export ra
# class_id lệch và validator sẽ báo lỗi.
DATA_YAML_CONTENT = """path: ./
train: train.txt
names:
  0: car
  1: motorcycle
  2: bus
  3: truck
"""


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Đóng gói ảnh + nhãn sơ bộ thành ZIP để import vào CVAT. "
            "Chỉ đóng gói, không sinh nhãn."
        )
    )
    parser.add_argument("--images", default="evaluation/images",
                        help="Thư mục ảnh cần gán nhãn.")
    parser.add_argument("--labels", default="evaluation/prelabels",
                        help="Thư mục nhãn sơ bộ. Để chuỗi rỗng nếu không dùng.")
    parser.add_argument("--output", default="evaluation/cvat_dataset.zip",
                        help="Đường dẫn file ZIP sinh ra.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Cho phép ghi đè file ZIP đã có.")
    return parser


def main():
    arguments = build_parser().parse_args()

    images_dir = Path(arguments.images)
    output_zip = Path(arguments.output)
    labels_dir = Path(arguments.labels) if arguments.labels else None

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")

    if labels_dir is not None and not labels_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục nhãn: {labels_dir}")

    if output_zip.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"File {output_zip} đã tồn tại. Thêm --overwrite nếu muốn ghi đè."
        )

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"Không có ảnh nào trong {images_dir}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    labels_written = 0
    labels_missing = []

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", DATA_YAML_CONTENT)

        # train.txt liệt kê đường dẫn tương đối của từng ảnh, đúng thứ tự.
        train_list = "\n".join(
            f"images/train/{path.name}" for path in image_paths
        )
        archive.writestr("train.txt", train_list + "\n")

        for image_path in image_paths:
            archive.write(image_path, f"images/train/{image_path.name}")

            if labels_dir is None:
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                archive.write(label_path, f"labels/train/{label_path.name}")
                labels_written += 1
            else:
                # Ảnh không có nhãn sơ bộ vẫn phải có file .txt rỗng, nếu không
                # CVAT có thể bỏ qua ảnh đó khi import.
                archive.writestr(f"labels/train/{image_path.stem}.txt", "")
                labels_missing.append(image_path.stem)

    size_mb = output_zip.stat().st_size / (1024 * 1024)

    print("ĐÃ TẠO ZIP CHO CVAT")
    print("=" * 60)
    print(f"  File            : {output_zip}")
    print(f"  Dung lượng      : {size_mb:.1f} MB")
    print(f"  Số ảnh          : {len(image_paths)}")
    if labels_dir is None:
        print(f"  Nhãn sơ bộ      : không kèm (vẽ từ đầu)")
    else:
        print(f"  Nhãn sơ bộ      : {labels_written}/{len(image_paths)} ảnh")
    if labels_missing:
        print(f"  Ảnh chưa có nhãn: {len(labels_missing)} "
              f"(đã tạo file rỗng): {labels_missing[:3]}")
    print()
    print("CÁC BƯỚC TRONG CVAT")
    print("=" * 60)
    print("  1. Tạo project mới, khai báo đúng 4 lớp theo thứ tự:")
    print("     car, motorcycle, bus, truck")
    print("  2. Tạo task và import file ZIP này")
    print("     (định dạng: Ultralytics YOLO Detection).")
    print("  3. Mở TỪNG ảnh và sửa: box thiếu, box thừa, sai lớp, box lệch.")
    print("  4. Export lại dạng Ultralytics YOLO Detection.")
    print("  5. Chép labels/train/*.txt vào evaluation/labels/.")
    print("  6. Kiểm tra:")
    print("     python evaluate_yolo.py --validate-only")
    print("     python tools/visualize_ground_truth.py --labels evaluation/labels")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print()
        print("LỖI:", error, file=sys.stderr)
        sys.exit(1)
