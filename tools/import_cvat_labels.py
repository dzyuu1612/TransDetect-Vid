"""
tools/import_cvat_labels.py
-----------------------------
Nhập nhãn đã kiểm tra thủ công từ CVAT vào repo, có REMAP class ID.

Vì sao bắt buộc phải remap thay vì chép thẳng:
    CVAT đánh số lớp theo thứ tự nhãn được khai báo trong Project, còn evaluator
    của repo dùng thứ tự cố định trong `evaluation/classes.txt`:

        evaluator: 0=car, 1=motorcycle, 2=bus, 3=truck

    Hai thứ tự này rất dễ lệch nhau. Ví dụ nếu CVAT khai báo
    `0=motorcycle, 1=car` thì chép thẳng file .txt sẽ biến mọi ô tô thành xe máy
    và ngược lại. Validator KHÔNG bắt được lỗi này vì cả hai ID đều nằm trong
    khoảng hợp lệ 0..3 — số liệu vẫn chạy ra nhưng sai hoàn toàn.

    Vì vậy script đọc trường `names` trong `data.yaml` của chính file export,
    ánh xạ ID cũ sang TÊN lớp, rồi từ tên lớp sang ID của evaluator.

Script ghi vào `evaluation/labels_candidate/`, KHÔNG ghi thẳng vào
`evaluation/labels/`. Bản candidate phải qua validator trước.

Cách chạy:
    python tools/import_cvat_labels.py --zip evaluation/cvat_ground_truth.zip
"""

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Thứ tự bắt buộc của evaluator, khớp evaluation/classes.txt.
EVALUATOR_CLASS_ID = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3}

# Chỉ chấp nhận alias đã biết; lớp lạ phải dừng để người dùng quyết định.
CLASS_ALIASES = {"motorbike": "motorcycle"}


def parse_names_from_data_yaml(text):
    """Đọc trường `names` trong data.yaml, trả về dict {id_cũ: tên lớp}.

    Hỗ trợ cả hai kiểu Ultralytics hay gặp:

        names:            và      names: [motorcycle, car, bus, truck]
          0: motorcycle
          1: car

    Tự phân tích bằng tay thay vì cài PyYAML: file data.yaml của CVAT rất đơn
    giản, thêm một thư viện chỉ để đọc bốn dòng là không cần thiết.
    """
    names = {}
    in_names_block = False

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        # Kiểu danh sách một dòng: names: [a, b, c, d]
        if line.startswith("names:") and "[" in line:
            inside = line.split("[", 1)[1].rsplit("]", 1)[0]
            for index, item in enumerate(inside.split(",")):
                cleaned = item.strip().strip("'\"")
                if cleaned:
                    names[index] = cleaned
            return names

        # Kiểu khối nhiều dòng: names: rồi các dòng "  0: car"
        if line.startswith("names:"):
            in_names_block = True
            continue

        if in_names_block:
            # Khối kết thúc khi gặp một khoá khác ở cột 0.
            if not raw_line.startswith((" ", "\t")):
                in_names_block = False
                continue

            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().strip("'\"")
            value = value.strip().strip("'\"")
            if key.isdigit() and value:
                names[int(key)] = value

    return names


def build_class_id_mapping(cvat_names):
    """Dựng bảng ánh xạ {id_CVAT: id_evaluator} dựa trên TÊN lớp.

    Ném ValueError nếu gặp lớp không thuộc bốn lớp đánh giá — không được âm
    thầm bỏ qua, vì như vậy sẽ mất ground-truth box và làm Recall sai.

    Trả về (mapping, danh sách dòng mô tả để in ra).
    """
    mapping = {}
    rows = []

    for cvat_id in sorted(cvat_names):
        raw_name = cvat_names[cvat_id]
        name = raw_name.strip().lower()
        alias_note = ""

        if name in CLASS_ALIASES:
            alias_note = f" (alias của {name})"
            name = CLASS_ALIASES[name]

        if name not in EVALUATOR_CLASS_ID:
            raise ValueError(
                f"Lớp '{raw_name}' (id {cvat_id} trong CVAT) không thuộc bốn lớp "
                f"đánh giá {sorted(EVALUATOR_CLASS_ID)}.\n"
                f"Dừng lại để bạn quyết định: đổi tên lớp trong CVAT và export "
                f"lại, hoặc xác nhận bỏ lớp này."
            )

        mapping[cvat_id] = EVALUATOR_CLASS_ID[name]
        rows.append((cvat_id, raw_name + alias_note, EVALUATOR_CLASS_ID[name]))

    return mapping, rows


def remap_label_text(text, mapping, source_name):
    """Đổi class_id của từng dòng nhãn, giữ nguyên bốn toạ độ.

    Toạ độ được ghi lại nguyên văn chuỗi gốc, không parse thành float rồi format
    lại — làm vậy sẽ thêm sai số làm tròn không cần thiết vào ground truth.
    """
    output_lines = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()

        if len(parts) == 6:
            raise ValueError(
                f"{source_name}:{line_number} có 6 trường. Nhiều khả năng bạn đã "
                f"export nhầm định dạng 'Detection Track' (trường thứ 6 là track "
                f"ID) thay vì 'Detection'. Hãy export lại đúng "
                f"'Ultralytics YOLO Detection 1.0'."
            )

        if len(parts) != 5:
            raise ValueError(
                f"{source_name}:{line_number} có {len(parts)} trường, cần đúng 5 "
                f"(class_id x_center y_center width height): {line!r}"
            )

        try:
            old_id = int(parts[0])
        except ValueError:
            raise ValueError(
                f"{source_name}:{line_number} class_id không phải số nguyên: "
                f"{parts[0]!r}"
            )

        if old_id not in mapping:
            raise ValueError(
                f"{source_name}:{line_number} class_id={old_id} không có trong "
                f"data.yaml của file export (chỉ có {sorted(mapping)})."
            )

        output_lines.append(f"{mapping[old_id]} {' '.join(parts[1:5])}")

    return output_lines


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Nhập nhãn CVAT đã kiểm tra thủ công, remap class ID theo data.yaml, "
            "ghi vào evaluation/labels_candidate/."
        )
    )
    parser.add_argument("--zip", default="evaluation/cvat_ground_truth.zip",
                        help="File ZIP export từ CVAT.")
    parser.add_argument("--images", default="evaluation/images",
                        help="Thư mục ảnh để đối chiếu tên file.")
    parser.add_argument("--extract-to",
                        default="evaluation/cvat_ground_truth_extracted",
                        help="Thư mục giải nén trung gian.")
    parser.add_argument("--output", default="evaluation/labels_candidate",
                        help="Thư mục ghi nhãn đã remap.")
    return parser


def main():
    arguments = build_parser().parse_args()

    zip_path = Path(arguments.zip)
    images_dir = Path(arguments.images)
    extract_dir = Path(arguments.extract_to)
    output_dir = Path(arguments.output)

    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {zip_path}.\n"
            f"Hãy export từ CVAT theo định dạng 'Ultralytics YOLO Detection 1.0' "
            f"và đặt file vào đúng đường dẫn này."
        )
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")

    with zipfile.ZipFile(zip_path) as archive:
        entry_names = archive.namelist()

        # Chặn path traversal trước khi ghi bất cứ thứ gì ra đĩa.
        for name in entry_names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"ZIP chứa đường dẫn không an toàn: {name!r}")

        yaml_entries = [n for n in entry_names
                        if n.lower().endswith(("data.yaml", "data.yml"))]
        if not yaml_entries:
            raise ValueError("ZIP không có data.yaml — không xác định được ánh xạ lớp.")

        data_yaml_text = archive.read(yaml_entries[0]).decode("utf-8")
        archive.extractall(extract_dir)

    cvat_names = parse_names_from_data_yaml(data_yaml_text)
    if not cvat_names:
        raise ValueError(
            f"Không đọc được trường `names` trong {yaml_entries[0]}."
        )

    mapping, mapping_rows = build_class_id_mapping(cvat_names)

    print("ÁNH XẠ CLASS ID")
    print("=" * 60)
    print(f"  data.yaml: {yaml_entries[0]}")
    print(f"  {'ID CVAT':>8}  {'Tên lớp':<24} {'ID evaluator':>12}")
    for cvat_id, name, evaluator_id in mapping_rows:
        changed = "  <-- ĐỔI" if cvat_id != evaluator_id else ""
        print(f"  {cvat_id:>8}  {name:<24} {evaluator_id:>12}{changed}")
    print()

    label_files = sorted(
        path for path in extract_dir.rglob("*.txt")
        if path.name.lower() != "train.txt"
    )
    if not label_files:
        raise ValueError(f"Không tìm thấy file nhãn .txt nào trong {extract_dir}")

    image_stems = {
        path.stem for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    boxes_per_class = {name: 0 for name in EVALUATOR_CLASS_ID}
    id_to_name = {value: key for key, value in EVALUATOR_CLASS_ID.items()}
    written_stems = set()
    empty_label_files = 0

    for label_path in label_files:
        lines = remap_label_text(
            label_path.read_text(encoding="utf-8"), mapping, label_path.name
        )

        (output_dir / f"{label_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        written_stems.add(label_path.stem)

        if not lines:
            empty_label_files += 1

        for line in lines:
            total_boxes += 1
            boxes_per_class[id_to_name[int(line.split()[0])]] += 1

    # Ảnh không có file nhãn trong ZIP: KHÔNG tự tạo file rỗng thay người dùng,
    # vì "frame không có xe" và "quên gán nhãn frame này" là hai chuyện khác
    # nhau, và chỉ người gán nhãn mới phân biệt được.
    missing = sorted(image_stems - written_stems)
    orphan = sorted(written_stems - image_stems)

    print("KẾT QUẢ NHẬP NHÃN")
    print("=" * 60)
    print(f"  Số ảnh                : {len(image_stems)}")
    print(f"  Số file nhãn đã ghi   : {len(written_stems)}")
    print(f"  Nhãn thiếu            : {len(missing)}")
    print(f"  Nhãn thừa             : {len(orphan)}")
    print(f"  File nhãn rỗng        : {empty_label_files}")
    print(f"  Tổng ground-truth box : {total_boxes}")
    for name in EVALUATOR_CLASS_ID:
        print(f"    {name:<12}: {boxes_per_class[name]}")
    print(f"  Thư mục candidate     : {output_dir}")

    if missing:
        print()
        print(f"  CẢNH BÁO: {len(missing)} ảnh không có nhãn: {missing[:5]}")
    if orphan:
        print()
        print(f"  CẢNH BÁO: {len(orphan)} nhãn không có ảnh: {orphan[:5]}")

    print()
    print("BƯỚC TIẾP THEO")
    print("=" * 60)
    print("  python evaluate_yolo.py --images evaluation/images "
          "--labels evaluation/labels_candidate --validate-only")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, zipfile.BadZipFile) as error:
        print()
        print("LỖI:", error, file=sys.stderr)
        sys.exit(1)
