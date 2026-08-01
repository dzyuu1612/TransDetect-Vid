"""
train_yolo.py
---------------
Tiện ích TÙY CHỌN để fine-tune hoặc kiểm định YOLO11 trên một dataset riêng.

File này KHÔNG cần thiết cho kết quả chính của đề tài: các chỉ số trong báo cáo
được sinh bằng `evaluate_yolo.py` chạy trên trọng số pretrained `yolo11n.pt`.
Chỉ dùng file này nếu nhóm thực sự huấn luyện lại model.

Vì sao không còn tự chạy khi mở file: bản trước liệt kê sẵn bốn dataset và huấn
luyện lần lượt ngay khi chạy `python train_yolo.py`, mỗi dataset 50 epoch. Ai
lỡ mở file là mất nhiều giờ GPU ngoài ý muốn. Ngoài ra vòng lặp cũ bọc
`except Exception` rồi in lỗi và chạy tiếp dataset kế tiếp, nên một lỗi đường
dẫn có thể trôi qua mà không ai nhận ra. Nay mỗi lần chạy chỉ làm đúng một việc
người dùng gõ ra, và lỗi được để nổi lên bình thường.

Cách chạy:
    python train_yolo.py train --data datasets/my_dataset/data.yaml \
        --model yolo11n.pt --epochs 50 --imgsz 640 --batch 16 \
        --name transdetect_run

    python train_yolo.py val --data datasets/my_dataset/data.yaml \
        --model runs/detect/transdetect_run/weights/best.pt --imgsz 640
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def train_model(data_yaml_path, model_path, epochs, img_size, batch_size,
                run_name, patience):
    """Huấn luyện YOLO11 trên dataset do người dùng chỉ định.

    `model_path` có thể là trọng số pretrained (`yolo11n.pt`) để fine-tune, hoặc
    một file `.pt` đã huấn luyện trước đó để học tiếp.

    Trả về đường dẫn thư mục kết quả THẬT do Ultralytics báo lại, thay vì ghép
    cứng chuỗi `runs/detect/<name>`: khi thư mục đã tồn tại, Ultralytics tự đổi
    sang `<name>2`, `<name>3`... nên đường dẫn ghép cứng sẽ trỏ sai chỗ.
    """
    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file data.yaml: {data_yaml_path}")

    model = YOLO(model_path)

    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=run_name,
        patience=patience,
        plots=True,
    )

    return Path(results.save_dir)


def validate_model(data_yaml_path, model_path, img_size):
    """Kiểm định một model đã huấn luyện trên tập validation của dataset.

    Lưu ý: chỉ dùng được khi `model_path` có đúng bộ lớp của `data.yaml`. KHÔNG
    gọi hàm này với trọng số COCO (`yolo11n.pt`) và một data.yaml bốn lớp — class
    ID của hai bên khác nhau nên số liệu vẫn chạy ra nhưng sai ý nghĩa. Muốn
    chấm trọng số COCO trên bốn lớp của đề tài thì dùng `evaluate_yolo.py`, ở đó
    việc ghép lớp đi qua TÊN lớp chứ không qua ID.
    """
    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file data.yaml: {data_yaml_path}")

    model = YOLO(model_path)
    metrics = model.val(data=str(data_yaml_path), imgsz=img_size)

    print()
    print("KẾT QUẢ KIỂM ĐỊNH")
    print("=" * 50)
    print(f"  mAP@0.50     : {metrics.box.map50:.4f}")
    print(f"  mAP@0.50:0.95: {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print("=" * 50)

    return metrics


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Tiện ích tùy chọn để huấn luyện / kiểm định YOLO11. "
            "Kết quả chính của đề tài dùng yolo11n.pt pretrained "
            "và evaluate_yolo.py."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Huấn luyện model.")
    train_parser.add_argument("--data", required=True,
                              help="Đường dẫn file data.yaml của dataset.")
    train_parser.add_argument("--model", default="yolo11n.pt",
                              help="Trọng số khởi đầu.")
    train_parser.add_argument("--epochs", type=int, default=50,
                              help="Số vòng huấn luyện.")
    train_parser.add_argument("--imgsz", type=int, default=640,
                              help="Kích thước ảnh đưa vào model.")
    train_parser.add_argument("--batch", type=int, default=16,
                              help="Số ảnh mỗi lần cập nhật trọng số.")
    train_parser.add_argument("--name", default="transdetect_run",
                              help="Tên thư mục kết quả trong runs/detect/.")
    train_parser.add_argument("--patience", type=int, default=15,
                              help="Số epoch không cải thiện thì dừng sớm.")

    val_parser = subparsers.add_parser("val", help="Kiểm định model đã train.")
    val_parser.add_argument("--data", required=True,
                            help="Đường dẫn file data.yaml của dataset.")
    val_parser.add_argument("--model", required=True,
                            help="Đường dẫn file .pt cần kiểm định.")
    val_parser.add_argument("--imgsz", type=int, default=640,
                            help="Kích thước ảnh đưa vào model.")

    return parser


def main():
    arguments = build_parser().parse_args()

    if arguments.command == "train":
        save_directory = train_model(
            data_yaml_path=arguments.data,
            model_path=arguments.model,
            epochs=arguments.epochs,
            img_size=arguments.imgsz,
            batch_size=arguments.batch,
            run_name=arguments.name,
            patience=arguments.patience,
        )
        print()
        print(f"Huấn luyện xong. Thư mục kết quả: {save_directory}")
        print(f"Trọng số tốt nhất: {save_directory / 'weights' / 'best.pt'}")
        return 0

    validate_model(
        data_yaml_path=arguments.data,
        model_path=arguments.model,
        img_size=arguments.imgsz,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileNotFoundError as error:
        print("LỖI:", error, file=sys.stderr)
        sys.exit(1)
