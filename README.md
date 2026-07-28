# TransDetect-Vid — Phát hiện phương tiện trong video

Hệ thống phát hiện phương tiện giao thông (ô tô, xe máy, xe buýt, xe tải)
trong video theo hai hướng tiếp cận song song:

1. **Pipeline truyền thống** — các phép biến đổi ảnh thủ công: cân bằng
   histogram, lọc trung vị, ngưỡng hoá lặp, Sobel, trích contour, kèm
   Lucas-Kanade Optical Flow để trực quan hoá chuyển động.
2. **Pipeline học sâu** — YOLO11n (Ultralytics) phát hiện và phân loại
   phương tiện trực tiếp trên từng khung hình.

Có cả giao diện dòng lệnh (CLI) và dashboard web (Streamlit).

---

## Cấu trúc mã nguồn

### Module lõi — `src/transdetect/`

Đây là mã nguồn chính, đúng những gì báo cáo trích trong Listing 3.1–3.4.

| File | Vai trò |
|---|---|
| `config.py` | Toàn bộ tham số tập trung một chỗ (ngưỡng, kernel, conf/IoU) |
| `preprocessing.py` | Ảnh xám → cân bằng histogram → lọc trung vị |
| `classical_detector.py` | Ngưỡng hoá lặp + Sobel + contour → vùng ứng viên |
| `optical_flow.py` | `LucasKanadeTracker` — theo dõi điểm đặc trưng thưa |
| `yolo_detector.py` | `Yolo11VehicleDetector` — bọc Ultralytics, lọc lớp phương tiện |
| `visualization.py` | Vẽ box, nhãn và vector chuyển động lên khung hình |
| `pipeline.py` | Gom hai nhánh thành luồng xử lý video hoàn chỉnh |

### Điểm chạy

| File | Vai trò |
|---|---|
| `main.py` | CLI — gọi `src/transdetect/pipeline.py` |
| `app_streamlit.py` | Dashboard web, dùng trực tiếp các module lõi |
| `train_yolo.py` | Huấn luyện / đánh giá YOLO11 trên dataset tuỳ chỉnh |

### Đánh giá định lượng

| File | Vai trò |
|---|---|
| `prepare_evaluation_frames.py` | Trích N frame phân bố đều từ video |
| `evaluate_pipelines.py` | Chấm điểm Precision/Recall/F1 cho cả hai pipeline |
| `evaluation/` | Ảnh, nhãn ground truth và kết quả (xem `evaluation/README.md`) |

### File cũ, không dùng cho báo cáo

`app.py`, `yolo_detector.py` (ở thư mục gốc) và `run_demo.py` là bản trước
khi refactor, giữ lại để tham khảo. Chúng **không** được `main.py` hay
`app_streamlit.py` import, và cài đặt khác với `src/transdetect/`.

---

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Sử dụng

### Dashboard web

```bash
streamlit run app_streamlit.py
```

Cho phép chọn video, chọn phương pháp, chỉnh ngưỡng conf/IoU, xem FPS thời
gian thực và xuất kết quả ra CSV/JSON.

### Dòng lệnh

```bash
# Nhánh truyền thống (có vẽ mũi tên Lucas-Kanade)
python main.py --input test_videos/video.mp4 --method classical \
    --output outputs/classical_out.mp4

# Nhánh YOLO11
python main.py --input test_videos/video.mp4 --method yolo \
    --model yolo11n.pt --conf 0.25 --iou 0.45 --max-det 300 \
    --output outputs/yolo_out.mp4
```

Bỏ trống `--conf/--iou/--max-det` thì giá trị lấy từ
`src/transdetect/config.py`, nên chỉ có một nguồn chân lý cho mặc định.

---

## Quy trình đánh giá định lượng

So sánh hai pipeline bằng Precision, Recall và F1 tại ngưỡng IoU ≥ 0,5, đánh
giá **class-agnostic** (gộp Car/Motorcycle/Bus/Truck thành `vehicle`). Phải
gộp lớp vì pipeline truyền thống chỉ sinh vùng ứng viên, không phân loại
được loại xe — so sánh có phân biệt lớp sẽ đo nhầm khả năng *phân loại* thay
vì khả năng *định vị*.

### Bước 1 — Kiểm thử code (không cần dữ liệu)

```bash
python -m compileall .
python evaluate_pipelines.py --selftest
```

Kết quả mong đợi: `23/23 KIỂM THỬ ĐỀU PASS`.

### Bước 2 — Trích frame phân bố đều

```bash
python prepare_evaluation_frames.py --video "duong_dan/video.mp4" --count 100
```

Sinh ảnh vào `evaluation/images/` và ghi `evaluation/frame_manifest.csv`
(ảnh nào ứng với frame nào, ở giây thứ mấy).

### Bước 3 — Gán nhãn thủ công

Dùng LabelImg, CVAT hoặc Roboflow Annotate. Xuất định dạng YOLO vào
`evaluation/labels/`, mỗi ảnh một file `.txt` cùng tên (kể cả file rỗng nếu
frame không có phương tiện). Chỉ gán 4 lớp trong `evaluation/classes.txt`.

**Không dùng dự đoán của YOLO làm ground truth** — nếu lấy output của model
làm chuẩn rồi chấm điểm chính model đó, kết quả thành vòng lặp tự khẳng định
và vô nghĩa.

### Bước 4 — Kiểm tra nhãn

```bash
python evaluate_pipelines.py --validate-only
```

Kiểm tra 12 điều kiện (thiếu nhãn, sai `class_id`, toạ độ ngoài `[0,1]`,
box suy biến…) mà không nạp model. Chỉ đi tiếp khi không còn lỗi.

### Bước 5 — Chạy đánh giá

```bash
python evaluate_pipelines.py
```

Kết quả tại `evaluation/results/`:

| File | Nội dung |
|---|---|
| `summary_metrics.csv` | TP/FP/FN, Precision, Recall, F1 của từng pipeline |
| `per_frame_metrics.csv` | Số liệu chi tiết theo từng khung hình |
| `run_metadata.json` | Commit, SHA256 model, mọi tham số, phiên bản thư viện |

`run_metadata.json` là bằng chứng để tái lập: nếu sau này thư viện đổi mặc
định, file này cho biết kết quả cũ sinh ra dưới cấu hình nào.

---

## Lưu ý

- `yolo11n.pt` (bản Nano) nhẹ, chạy được cả trên CPU.
- Pipeline truyền thống giữ lại với mục đích giải thích: quan sát được từng
  phép biến đổi trung gian. Nó **không** phân loại được loại xe và **không**
  duy trì ID phương tiện qua các khung hình.
- Lucas-Kanade trong dự án chỉ theo dõi và trực quan hoá điểm đặc trưng —
  chưa gán điểm vào bounding box, chưa phải multi-object tracking.
- Các bộ đếm trên dashboard là số detection **theo từng khung hình**, không
  phải tổng số phương tiện duy nhất đi qua video.
