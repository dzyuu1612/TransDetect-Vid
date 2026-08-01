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
| `train_yolo.py` | **Tuỳ chọn** — fine-tune / kiểm định YOLO11 trên dataset riêng |

### Đánh giá định lượng

| File | Vai trò |
|---|---|
| `prepare_evaluation_frames.py` | Trích N frame phân bố đều từ video |
| `evaluate_yolo.py` | **Chỉ số chính** — chấm YOLO11 class-aware: P/R/F1, AP50, mAP50-95 |
| `evaluate_pipelines.py` | Tuỳ chọn — so sánh khả năng *định vị* class-agnostic của hai pipeline |
| `evaluation/` | Ảnh, nhãn ground truth và kết quả (xem `evaluation/README.md`) |

### Dữ liệu và tài liệu kèm theo

| Thư mục | Vai trò |
|---|---|
| `datasets/` | Nguồn và giấy phép của dataset Roboflow (chỉ README + `data.yaml`, không commit ảnh) |
| `notebooks/` | Notebook demo chạy trên Google Colab |
| `evaluation/` | Tập đánh giá và kết quả — xem `evaluation/README.md` |

> Các bản script trước khi refactor (`app.py`, `yolo_detector.py` ở thư mục
> gốc, `run_demo.py`) đã được gỡ bỏ: chúng trùng chức năng với
> `src/transdetect/` nhưng cài đặt khác, và không được `main.py` hay
> `app_streamlit.py` import. Cần xem lại thì tra trong lịch sử Git.

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

**Chỉ nhánh YOLO11 được đánh giá định lượng.** YOLO11 sinh bounding box kèm
nhãn lớp và confidence nên tính được Precision/Recall/F1 có phân biệt lớp và
dựng được đường Precision-Recall để tính AP/mAP. Nhánh truyền thống chỉ sinh
vùng ứng viên từ threshold/Sobel/contour: không có lớp và không có confidence
chuẩn để xếp hạng prediction, nên **không báo cáo mAP cho nhánh này**. Ép nó có
mAP bằng một confidence giả (diện tích contour, hằng số 1,0…) sẽ ra con số chạy
được nhưng sai ý nghĩa.

Nhánh truyền thống vẫn giữ để minh hoạ chuỗi phép biến đổi và **đo FPS** —
tức là đánh giá tốc độ và tính giải thích được, không phải độ chính xác.

> Không viết "YOLO chính xác hơn pipeline truyền thống X%" khi pipeline truyền
> thống không được chấm trên cùng một chuẩn.

### Bước 1 — Kiểm thử code (không cần dữ liệu)

```bash
python -m compileall main.py app_streamlit.py evaluate_yolo.py src/transdetect
python evaluate_yolo.py --selftest
```

Kết quả mong đợi: `60/60 KIỂM THỬ ĐỀU PASS`.

### Bước 2 — Trích frame phân bố đều

```bash
python prepare_evaluation_frames.py --video "duong_dan/video.mp4" --count 100
```

Sinh ảnh vào `evaluation/images/` và ghi `evaluation/frame_manifest.csv`
(ảnh nào ứng với frame nào, ở giây thứ mấy).

### Bước 3 — Gán nhãn thủ công

Dùng LabelImg, CVAT hoặc Roboflow Annotate. Xuất định dạng YOLO vào
`evaluation/labels/`, mỗi ảnh một file `.txt` cùng tên (kể cả file rỗng nếu
frame không có phương tiện). Chỉ gán 4 lớp trong `evaluation/classes.txt`, và
`class_id` phải đúng vì phép đánh giá này là class-aware.

**Không dùng dự đoán của YOLO làm ground truth** — nếu lấy output của model làm
chuẩn rồi chấm điểm chính model đó, kết quả thành vòng lặp tự khẳng định, luôn
ra gần 1,00 và vô nghĩa. Được phép pre-label bằng YOLO cho nhanh, nhưng người
gán nhãn phải xem và sửa toàn bộ box, lớp, box thiếu và box thừa.

### Bước 4 — Kiểm tra nhãn

```bash
python evaluate_yolo.py --validate-only
```

Kiểm tra thiếu/thừa nhãn, sai `class_id`, toạ độ ngoài `[0,1]`, box suy biến…
mà không nạp model. Script **dừng hẳn và không sinh file kết quả nào** nếu
dataset chưa hợp lệ — sai lệch âm thầm nguy hiểm hơn nhiều so với việc báo lỗi.

### Bước 5 — Chạy đánh giá

```bash
python evaluate_yolo.py \
  --images evaluation/images --labels evaluation/labels \
  --model yolo11n.pt --imgsz 640 --conf 0.25 --predict-conf 0.001 \
  --nms-iou 0.45 --match-iou 0.50 --output evaluation/results_yolo
```

Kết quả tại `evaluation/results_yolo/`:

| File | Nội dung |
|---|---|
| `summary_metrics.csv` | Một dòng tổng: TP/FP/FN, P/R/F1, mAP50, mAP50-95 |
| `per_class_metrics.csv` | Bốn lớp: GT, TP/FP/FN, P/R/F1, AP50, AP50-95 |
| `per_frame_metrics.csv` | Từng ảnh: GT, số prediction, TP/FP/FN |
| `predictions.csv` | Mọi prediction kèm confidence và toạ độ |
| `run_metadata.json` | Commit, SHA256 model, mọi ngưỡng, phiên bản thư viện |

`run_metadata.json` là bằng chứng để tái lập: nếu sau này thư viện đổi mặc
định, file này cho biết kết quả cũ sinh ra dưới cấu hình nào.

### Các chỉ số và ngưỡng

| Chỉ số | Ý nghĩa |
|---|---|
| Precision | Trong các box YOLO dự đoán, tỉ lệ box đúng |
| Recall | Trong các xe thật, tỉ lệ được YOLO tìm thấy |
| F1 | Cân bằng Precision/Recall tại `conf=0.25` |
| mAP@0.5 | AP trung bình tại IoU 0,50 |
| **mAP@0.5:0.95** | AP trung bình trên IoU 0,50→0,95 — **chỉ số chính** |

Confidence trung bình của box **không phải** độ chính xác; chỉ dùng để mô tả.

Hai ngưỡng IoU có vai trò khác nhau, không được nhầm: **NMS IoU = 0,45** dùng
để loại prediction trùng nhau trong lúc suy luận, còn **matching IoU = 0,50**
dùng để quyết định prediction có khớp ground truth hay không.

Suy luận chạy **một lượt duy nhất** ở `conf=0.001` rồi lọc lại ở `0.25`: AP cần
cả prediction điểm thấp ở phần đuôi để dựng trọn đường Precision-Recall, nếu chỉ
chạy ở 0,25 thì đường PR bị cắt cụt và AP thấp hơn giá trị thật.

`imgsz=640` là kích thước model nhận **sau letterbox**, không phải độ phân giải
video nguồn (1280×720).

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
