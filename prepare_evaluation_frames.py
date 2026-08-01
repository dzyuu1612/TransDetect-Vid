"""
prepare_evaluation_frames.py
------------------------------
Trích N khung hình PHÂN BỐ ĐỀU từ một video để làm tập đánh giá.

Vì sao phải phân bố đều mà không lấy N frame liên tiếp: các frame liên tiếp
trong video 59,94 FPS gần như giống hệt nhau (xe chỉ dịch 1-2 pixel). Nếu lấy
100 frame liên tiếp thì thực chất chỉ đánh giá trên khoảng 1,7 giây video và
gần như cùng một cảnh — kết quả sẽ không đại diện cho toàn bộ video.

Script này CHỈ trích ảnh. Nó KHÔNG chạy YOLO và KHÔNG sinh nhãn tự động, vì
nhãn ground truth bắt buộc phải do con người gán. Nếu dùng dự đoán của YOLO
làm nhãn chuẩn rồi chấm điểm chính YOLO, kết quả sẽ thành vòng lặp tự khẳng
định và hoàn toàn vô nghĩa.

Cách chạy:
    python prepare_evaluation_frames.py --video "duong_dan/video.mp4" \
        --count 100 --output evaluation/images \
        --manifest evaluation/frame_manifest.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


# Console Windows mặc định dùng cp1252, không mã hoá được tiếng Việt có dấu nên
# một lệnh print bình thường cũng làm script dừng giữa chừng bằng
# UnicodeEncodeError — dù frame đã trích xong. Ép stdout/stderr về UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_video_metadata(capture, video_path):
    """Đọc và kiểm tra metadata của video.

    Trả về dict gồm tổng số frame, FPS, độ phân giải và thời lượng. Ném lỗi
    nếu video không có frame nào — trường hợp này thường do file hỏng hoặc
    thiếu codec, tốt nhất là dừng ngay thay vì để script chạy tiếp rồi thất
    bại khó hiểu ở bước sau.
    """
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        raise ValueError(
            f"Video không có frame nào hoặc metadata hỏng: {video_path}\n"
            f"Kiểm tra lại file video và codec."
        )

    # fps có thể bằng 0 khi metadata hỏng; khi đó không tính được thời gian
    # nhưng vẫn trích được frame, nên chỉ cảnh báo chứ không dừng.
    duration_seconds = total_frames / fps if fps > 0 else 0.0

    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


def choose_frame_indices(total_frames, count):
    """Chọn `count` chỉ số frame phân bố đều trên toàn bộ video.

    Dùng np.linspace để chia đều khoảng [0, total_frames - 1]. Ví dụ video có
    780 frame và cần 100 frame thì bước nhảy khoảng 7,9 — linspace tự xử lý
    phần thập phân này, khác với việc dùng range(0, total, step) vốn có thể
    bỏ sót phần cuối video khi total không chia hết cho step.

    np.unique vừa loại trùng vừa sắp xếp tăng dần. Trùng chỉ xảy ra khi count
    gần bằng total_frames (linspace làm tròn hai giá trị liền nhau về cùng
    một số nguyên); khi đó số frame trả về sẽ ít hơn count và hàm gọi phải
    báo lại cho người dùng biết.
    """
    raw_indices = np.linspace(0, total_frames - 1, num=count)
    return np.unique(raw_indices.astype(int))


def extract_frames(video_path, count, output_directory, manifest_path,
                   overwrite=False):
    """Trích frame và ghi manifest.

    Manifest ghi lại frame nào trong video gốc tương ứng với ảnh nào, để sau
    này có thể truy ngược: một kết quả bất thường ở ảnh X đến từ giây thứ mấy
    của video. Không có manifest thì tên file `frame_000123.jpg` là thông tin
    duy nhất còn lại, và nó sẽ mất ý nghĩa nếu đổi tham số count.
    """
    video_path = Path(video_path)
    output_directory = Path(output_directory)
    manifest_path = Path(manifest_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    if count < 1:
        raise ValueError(f"--count phải >= 1, đang nhận: {count}")

    # Chặn ghi đè ngoài ý muốn: nếu thư mục đã có ảnh mà người dùng chưa nói
    # rõ là muốn ghi đè thì dừng lại. Trích lại frame với count khác sẽ tạo
    # ra bộ ảnh mới không khớp với các file nhãn đã gán công sức trước đó.
    if output_directory.is_dir():
        existing = [
            path for path in output_directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if existing and not overwrite:
            raise FileExistsError(
                f"Thư mục {output_directory} đã có {len(existing)} ảnh.\n"
                f"Trích lại sẽ tạo bộ ảnh mới không khớp với nhãn đã gán. "
                f"Thêm --overwrite nếu thực sự muốn ghi đè."
            )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")

    try:
        metadata = read_video_metadata(capture, video_path)
        total_frames = metadata["total_frames"]

        if count > total_frames:
            raise ValueError(
                f"--count ({count}) lớn hơn tổng số frame của video "
                f"({total_frames}). Giảm --count xuống."
            )

        frame_indices = choose_frame_indices(total_frames, count)
        if len(frame_indices) < count:
            print(
                f"  Lưu ý: chỉ chọn được {len(frame_indices)} chỉ số frame "
                f"khác nhau (yêu cầu {count}), do count quá gần tổng số frame."
            )

        output_directory.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        for frame_index in frame_indices:
            # CAP_PROP_POS_FRAMES cho phép nhảy thẳng tới frame cần lấy, thay
            # vì đọc tuần tự từ đầu. Với video dài thì đây là khác biệt lớn
            # về thời gian chạy.
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()

            # Không bỏ qua âm thầm frame lỗi: nếu thiếu frame mà vẫn chạy
            # tiếp, tập đánh giá sẽ ít hơn số đã ghi trong báo cáo mà không
            # ai biết.
            if not ok or frame is None:
                raise RuntimeError(
                    f"Không đọc được frame số {frame_index} của video. "
                    f"File có thể bị hỏng ở đoạn này."
                )

            image_name = f"frame_{int(frame_index):06d}.jpg"
            cv2.imwrite(str(output_directory / image_name), frame)

            time_seconds = (
                frame_index / metadata["fps"] if metadata["fps"] > 0 else 0.0
            )
            manifest_rows.append({
                "image_name": image_name,
                "source_frame_index": int(frame_index),
                "time_seconds": round(float(time_seconds), 3),
                "source_video": video_path.name,
                "source_width": metadata["width"],
                "source_height": metadata["height"],
                "source_fps": round(metadata["fps"], 2),
            })

        with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)

        return metadata, manifest_rows

    finally:
        # Đặt trong finally để video luôn được giải phóng kể cả khi có lỗi ở
        # giữa chừng; trên Windows, không release sẽ khoá file.
        capture.release()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trích N frame phân bố đều từ video để làm tập đánh giá. "
            "KHÔNG sinh nhãn tự động — nhãn phải do con người gán."
        )
    )
    parser.add_argument("--video", required=True, help="Đường dẫn video nguồn.")
    parser.add_argument(
        "--count", type=int, default=100,
        help="Số frame cần trích (mặc định 100).",
    )
    parser.add_argument(
        "--output", default="evaluation/images",
        help="Thư mục lưu ảnh.",
    )
    parser.add_argument(
        "--manifest", default="evaluation/frame_manifest.csv",
        help="Đường dẫn file manifest.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Cho phép ghi đè ảnh đã có trong thư mục output.",
    )
    args = parser.parse_args()

    metadata, rows = extract_frames(
        video_path=args.video,
        count=args.count,
        output_directory=args.output,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
    )

    print("THÔNG TIN VIDEO NGUỒN")
    print("=" * 60)
    print(f"  File          : {Path(args.video).name}")
    print(f"  Độ phân giải  : {metadata['width']}×{metadata['height']}")
    print(f"  FPS nguồn     : {metadata['fps']:.2f}")
    print(f"  Tổng số frame : {metadata['total_frames']}")
    print(f"  Thời lượng    : {metadata['duration_seconds']:.2f} giây")
    print()
    print("KẾT QUẢ TRÍCH FRAME")
    print("=" * 60)
    print(f"  Số frame đã trích : {len(rows)}")
    print(f"  Thư mục ảnh       : {args.output}")
    print(f"  Manifest          : {args.manifest}")
    print()
    print("BƯỚC TIẾP THEO")
    print("=" * 60)
    print("  1. Gán nhãn thủ công bằng CVAT / Roboflow Annotate.")
    print("     car=0, motorcycle=1, bus=2, truck=3.")
    print("  2. Lưu nhãn YOLO vào evaluation/labels/.")
    print("     Mỗi ảnh phải có một file .txt cùng tên.")
    print("  3. Kiểm tra evaluator YOLO11:")
    print("     python evaluate_yolo.py --validate-only")
    print("  4. Chạy đánh giá chính YOLO11:")
    print("     python evaluate_yolo.py")
    print("  5. So sánh định vị hai pipeline, nếu cần:")
    print("     python evaluate_pipelines.py --output evaluation/results_compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
