# CHANGELOG

Ghi lại các thay đổi theo từng đợt làm việc. Mục đích là để đối chiếu ngược
giữa mã nguồn và báo cáo: mỗi con số trong báo cáo phải truy được về đúng
commit, đúng lệnh chạy và đúng file kết quả.

---

## [2026-08-02] — Đánh giá theo ROI và hoàn thiện repo

### Thêm mới

- **`--roi-y-min`** cho cả `evaluate_yolo.py` và `evaluate_pipelines.py` — giới
  hạn phạm vi đánh giá theo TỈ LỆ chiều cao ảnh, lọc theo TÂM bounding box.
  Xét tâm chứ không xét mép vì xe vắt qua biên vẫn phải được phân loại dứt
  khoát, và ground truth với prediction hiếm khi trùng mép nhau.
  - Lọc đặt trong `predict_all_images` — sớm nhất trong luồng dữ liệu — nên
    P/R/F1, AP50, mAP50-95, per-frame và per-class đều cùng một phạm vi. Nếu chỉ
    lọc ở `conf=0.25` thì mAP vẫn là full-frame và hai nhóm chỉ số không so sánh
    được với nhau.
  - Số GT trong CSV đếm từ tập ĐÃ lọc, không lấy từ `validate_dataset` (vốn luôn
    là full-frame), nếu không đẳng thức `TP + FN = GT` sẽ vỡ.
  - Mặc định `None`: regression đã xác nhận 4 file CSV trùng khít từng byte với
    kết quả full-frame trước đó.

- **`tools/visualize_roi_ground_truth.py`** — vẽ đường biên ROI làm bằng chứng
  cho phạm vi công bố. Box ngoài ROI vẽ xám mảnh chứ không ẩn, để không gây hiểu
  nhầm là tập nhãn vốn không có chúng.

- **`tools/import_cvat_labels.py`** — nhập nhãn CVAT có remap class ID theo
  `data.yaml` của chính file export. CVAT dùng `0=motorcycle, 1=car` còn
  evaluator dùng `0=car, 1=motorcycle`; chép thẳng sẽ biến mọi ô tô thành xe máy
  mà validator không bắt được vì cả hai ID đều hợp lệ.

- **`evaluation/ANNOTATION_AND_ROI_LIMITATIONS.md`** — ghi lại giới hạn khoa học.

### Dữ liệu

- Nhập 100 file nhãn từ CVAT (3.717 box full-frame; 904 box trong ROI).
- Audit phát hiện **99,5% box giữ nguyên toạ độ pre-label do YOLO11 sinh ra**,
  và 99,6% GT box có IoU ≥ 0,999 với một prediction của cùng mô hình. Đây là
  nguyên nhân `mAP50 = mAP50-95 = 0,9950`: mọi cặp khớp có IoU ≈ 1,0 nên quét
  ngưỡng 0,50→0,95 không đổi gì. Precision và mAP vì thế lạc quan.

### Kết quả (có cảnh báo assisted annotation)

Trong ROI `center_y/H >= 0,60`, YOLO11n tại `imgsz=640`, `conf=0,25`,
matching IoU `0,50`: TP=696, FP=0, FN=208 → Precision 100,00%, Recall 76,99%,
**F1 87,00%**. Classical cùng ROI (class-agnostic): F1 1,01%.
Bus và Truck có 0 GT trong ROI nên ghi `N/A` và bị loại khỏi mAP.

### Sửa lỗi

- `tools/visualize_roi_ground_truth.py` ban đầu đếm 896 box trong ROI trong khi
  evaluator đếm 904. Nguyên nhân: preview làm tròn toạ độ về `int` để vẽ, dịch
  tâm box nửa pixel và đẩy 8 box sát biên sang phía bên kia. Đã đổi sang đếm
  trên toạ độ chuẩn hoá gốc; hai bên giờ khớp chính xác.

### Dọn dẹp

- Khôi phục 11 file metadata/giấy phép trong `datasets/` — đây là nguồn dữ liệu
  và điều khoản sử dụng, cần cho tái lập, không phải rác.
- Đổi tên kết quả full-frame cũ thành
  `evaluation/not_for_report_results_yolo_fullframe/` và ignore, để không nhầm
  với kết quả ROI chính thức.
- `.gitignore` bổ sung: ZIP CVAT, thư mục giải nén, `labels_candidate/`,
  các thư mục preview và `not_for_report_*/`.

### Self-test

`evaluate_yolo.py` 60/60 → **74/74**; `evaluate_pipelines.py` 23/23 → **30/30**.

---

## [Chưa phát hành] — Đánh giá class-aware cho YOLO11 và dọn repo

### Thêm mới

- **`evaluate_yolo.py`** — chấm điểm định lượng RIÊNG cho nhánh YOLO11.
  - **Class-aware**: một prediction chỉ tính TP khi trùng cả vị trí lẫn lớp.
    Khác `evaluate_pipelines.py` vốn bỏ `class_id` để chấm class-agnostic.
  - Ghép lớp qua **TÊN** đã chuẩn hoá, không qua ID. Đây là điểm dễ sai nhất:
    `yolo11n.pt` pre-train trên COCO dùng ID nội bộ (car=2, motorcycle=3,
    bus=5, truck=7), khác hoàn toàn ID của tập nhãn (0/1/2/3). So trực tiếp
    hai bộ ID thì mọi prediction đều lệch lớp — script vẫn ra số nhưng vô nghĩa.
  - Thêm **AP@0.50** và **AP@0.50:0.95** theo từng lớp, nội suy 101 điểm
    recall, ghép theo confidence giảm dần. Suy luận **một lượt duy nhất** ở
    `conf=0.001` rồi lọc lại ở `0.25` cho nhóm P/R/F1: AP cần cả prediction
    điểm thấp ở phần đuôi, chạy thẳng ở 0,25 sẽ cắt cụt đường PR và làm AP
    thấp hơn giá trị thật.
  - Lớp không có ground truth trả về `N/A`, **không** phải 0.0 — 0.0 sẽ bị
    tính vào trung bình và kéo mAP xuống sai lệch.
  - Lớp ngoài bốn lớp đánh giá (`person`, `bicycle`…) bị bỏ qua chứ không
    tính thành FP, vì tập nhãn không gán chúng.
  - Tách bạch **NMS IoU = 0,45** (loại prediction trùng lúc suy luận) và
    **matching IoU = 0,50** (quyết định khớp ground truth) thành hai tham số.
  - Sinh 5 file: `summary_metrics.csv`, `per_class_metrics.csv`,
    `per_frame_metrics.csv`, `predictions.csv`, `run_metadata.json`.
  - `--selftest`: **60/60 PASS**, không cần ảnh/nhãn/model.
  - `--validate-only`: dừng hẳn và **không sinh file nào** nếu dataset sai.

### Thay đổi

- **`src/transdetect/yolo_detector.py`** — thêm tham số `img_size=640` ở cuối
  `detect_frame()` và truyền vào `model.predict(imgsz=...)`. Cả 4 nơi đang gọi
  đều truyền positional tới `max_det`, và 640 đúng bằng giá trị Ultralytics vẫn
  dùng ngầm trước đây, nên hành vi demo không đổi.

- **`train_yolo.py`** — chuyển sang CLI có hai lệnh con `train` / `val`.
  Bản cũ liệt kê sẵn bốn dataset và huấn luyện lần lượt 50 epoch **ngay khi
  chạy file**, nên ai lỡ mở là mất nhiều giờ GPU ngoài ý muốn; vòng lặp còn
  bọc `except Exception` rồi chạy tiếp dataset kế, khiến lỗi đường dẫn trôi
  qua không ai biết. Nay in đường dẫn kết quả THẬT do Ultralytics trả về
  (`results.save_dir`) thay vì ghép cứng `runs/detect/<name>` — Ultralytics tự
  đổi sang `<name>2`, `<name>3`… khi thư mục đã tồn tại.

### Sửa lỗi

- **`prepare_evaluation_frames.py`** — script dừng bằng `UnicodeEncodeError`
  ở lệnh `print` cuối trên console Windows (cp1252 không mã hoá được tiếng
  Việt có dấu), *sau khi* đã trích xong frame. Ép `stdout`/`stderr` về UTF-8;
  áp dụng tương tự cho `evaluate_yolo.py`.

### Gỡ bỏ

- `app.py`, `yolo_detector.py` (thư mục gốc), `run_demo.py` — bản trước khi
  refactor, trùng chức năng với `src/transdetect/` nhưng cài đặt khác và không
  được `main.py` hay `app_streamlit.py` import.
- `CODE_WALKTHROUGH.md`, `DOCX_REVISIONS_v14.md`, `UI_REQUIREMENTS.md`,
  `HUONG_DAN_CAI_DAT.md`, `file_word_BaoCao/` — tài liệu và bản báo cáo đi kèm,
  không thuộc mã nguồn hay demo. Tra lại được trong lịch sử Git.

### Trạng thái

Chưa có số liệu nào để đưa vào báo cáo: `evaluation/labels/` vẫn rỗng. Toàn bộ
mã đánh giá đã xong và đã kiểm thử, chỉ còn thiếu phần nhãn do **con người**
gán — không được dùng output của YOLO làm ground truth.

---

## [Chưa phát hành] — Chuẩn bị dữ liệu và siết chặt kiểm tra

### Thêm mới

- **`prepare_evaluation_frames.py`** — trích N frame phân bố đều từ video.
  - Dùng `np.linspace` thay vì `range(0, total, step)`: linspace xử lý được
    phần thập phân của bước nhảy nên phủ trọn tới frame cuối, còn range có
    thể bỏ sót đoạn cuối video khi tổng số frame không chia hết cho bước.
  - Sinh `evaluation/frame_manifest.csv` ghi lại ảnh nào ứng với frame nào,
    ở giây thứ mấy — để sau này truy ngược được một kết quả bất thường.
  - Từ chối ghi đè khi thư mục đã có ảnh (trừ khi truyền `--overwrite`), vì
    trích lại với `--count` khác sẽ tạo bộ ảnh mới không khớp nhãn đã gán.
  - Báo lỗi và dừng nếu không đọc được một frame đã chọn, thay vì bỏ qua âm
    thầm làm tập đánh giá ít hơn số đã ghi trong báo cáo.
  - **Không** chạy YOLO và **không** sinh nhãn tự động.

- **`evaluation/classes.txt`** — cố định 4 lớp: `car=0, motorcycle=1,
  bus=2, truck=3`.

- **Đã trích 100 frame thật** từ `duong_pho_sg(3).mp4` (1280×720, 59,94 FPS,
  780 frame) vào `evaluation/images/`, bước nhảy 7–8 frame, phủ từ giây 0
  đến giây 12,996.

### Siết chặt `evaluate_pipelines.py`

- Thêm `validate_evaluation_dataset()` kiểm tra 12 điều kiện: thư mục tồn
  tại, ảnh đọc được, mọi ảnh có đúng một nhãn, không có nhãn thừa, mỗi dòng
  đủ 5 trường, `class_id` thuộc {0,1,2,3}, toạ độ hữu hạn và nằm trong
  `[0,1]`, `width`/`height` dương, box không suy biến sau khi đổi sang pixel.
  File nhãn rỗng được chấp nhận vì frame có thể thật sự không có xe.
- **Đổi hành vi**: ảnh thiếu nhãn trước đây chỉ *cảnh báo* rồi coi như 0
  ground truth — nay **dừng hẳn**. Lý do: mọi box phát hiện trên ảnh thiếu
  nhãn đều thành False Positive, chỉ vài ảnh là Precision đã sai lệch mà
  không để lại dấu vết nào trong CSV. Sai lệch âm thầm nguy hiểm hơn lỗi ồn ào.
- Thêm `--validate-only`: kiểm tra dataset mà không nạp model YOLO.
- Thêm `evaluation/results/run_metadata.json`: commit Git, SHA256 của file
  trọng số, toàn bộ ngưỡng, cấu hình Classical, thiết bị, và phiên bản
  Python/OpenCV/NumPy/Ultralytics/Torch **đọc từ môi trường thật**, không
  ghi cứng.
- Thêm kiểm tra bất biến sau khi chấm: `TP + FN` phải bằng tổng ground truth,
  và Precision/Recall/F1 phải nằm trong `[0, 1]`.
- Bắt `ValueError`/`FileNotFoundError` ở `__main__` để in thông báo gọn thay
  vì traceback dài, vì đây là lỗi dữ liệu của người dùng chứ không phải bug.

### Sửa lỗi trong tài liệu và thông báo

- **`23/23 PASS`, không phải `20/20`.** Bản CHANGELOG trước ghi sai do đếm
  nhầm số ca kiểm thử. Nay `run_selftest()` tự đếm động nên không thể lệch lại.
- Thông báo lỗi trong `evaluate_pipelines.py` trỏ tới một file đặc tả không
  có trong repo → đổi thành `evaluation/README.md`.
- Bảng kết quả định lượng là **Bảng 4.4**, không phải 4.5 (đã sửa các chỗ
  ghi nhầm trong CHANGELOG).

### Kiểm thử đợt này

| Hạng mục | Kết quả |
|---|---|
| `python -m compileall` | OK |
| `--selftest` | 23/23 PASS |
| Trích 100 frame từ video thật | OK, bước nhảy 7–8, phủ toàn video |
| Validator bắt lỗi nhãn | Bắt đúng 4/4 lỗi cố ý (thiếu trường, sai `class_id`, toạ độ ngoài `[0,1]`, `width=0`) kèm số dòng |
| Validator chấp nhận nhãn hợp lệ | OK, kể cả file nhãn rỗng |
| `--validate-only` | Chạy được, không nạp model |
| Chạy đầy đủ trên 6 ảnh + nhãn giả lập | Sinh đủ 3 file kết quả, metadata đầy đủ |

Dữ liệu giả lập dùng để kiểm thử đã bị xoá. `evaluation/labels/` và
`evaluation/results/` hiện **rỗng** — không có nhãn giả hay số liệu giả nào
lọt vào repo.

---

## [Chưa phát hành] — Bổ sung quy trình đánh giá định lượng

Đặc tả nguồn: tài liệu yêu cầu do nhóm cung cấp (không nằm trong repo)
Mã nguồn đối chiếu: nhánh `main`, commit `6920c6ac879c8e1c93a662a8dca2dda965c45563`

### Thêm mới

- **`evaluate_pipelines.py`** — script đánh giá độc lập, so sánh nhánh truyền
  thống và YOLO11 trên cùng tập frame có ground truth.
  - Chỉ *đọc lại* các hàm phát hiện sẵn có (`preprocessing.preprocess_frame`,
    `classical_detector.detect_vehicle_candidates`,
    `Yolo11VehicleDetector.detect_frame`), không sửa thuật toán.
  - So sánh **class-agnostic**: gộp Car/Motorcycle/Bus/Truck thành một lớp
    `vehicle`. Bắt buộc phải vậy vì nhánh truyền thống chỉ sinh vùng ứng viên,
    không phân loại được loại xe — nếu so sánh có phân biệt lớp thì nhánh này
    sẽ sai gần như tuyệt đối và kết quả không phản ánh đúng khả năng *định vị*
    phương tiện của nó.
  - Ghép prediction ↔ ground truth **một-một** theo IoU giảm dần, ngưỡng
    `IoU >= 0.5` mới tính True Positive. Ràng buộc một-một khiến hai box cùng
    chồng lên một chiếc xe chỉ được tính một TP, box thừa thành FP.
  - Tính TP/FP/FN → Precision, Recall, F1 (micro-average trên toàn tập, không
    lấy trung bình chỉ số của từng frame, để frame có 30 xe không bị cùng
    trọng số với frame có 1 xe).
  - Xuất `evaluation/results/summary_metrics.csv` và
    `evaluation/results/per_frame_metrics.csv`.
  - Cờ `--selftest` chạy kiểm thử các hàm IoU/matching/metric mà không cần
    dữ liệu, model hay ảnh.
  - Không thêm dependency: chỉ dùng thư viện chuẩn Python + `cv2` (đã có sẵn).

- **`evaluation/`** — cấu trúc thư mục `images/`, `labels/`, `results/` (hiện
  rỗng, chỉ có `.gitkeep`) kèm `evaluation/README.md` hướng dẫn cách trích
  frame phân bố đều, quy tắc gán nhãn và các lỗi cần tránh.

- **`CHANGELOG.md`** — file này.

### Không thay đổi

Theo yêu cầu, các file sau **không bị sửa một dòng nào**:

```text
main.py
app_streamlit.py
src/transdetect/preprocessing.py
src/transdetect/classical_detector.py
src/transdetect/optical_flow.py
src/transdetect/yolo_detector.py
src/transdetect/pipeline.py
```

### Kiểm thử

`python evaluate_pipelines.py --selftest` — **23/23 PASS**:

| Nhóm | Ca kiểm thử |
|---|---|
| `calculate_iou` | box trùng khít (=1), box rời nhau (=0), chồng một nửa (=1/3), box suy biến không lỗi chia 0, ví dụ Mục 2.6.5 của báo cáo (=0,5714) |
| `match_predictions` | khớp hoàn hảo, không prediction, không ground truth, **hai prediction cùng chồng lên một ground truth → đúng 1 TP + 1 FP**, chồng lấp dưới ngưỡng bị loại |
| `calculate_metrics` | ví dụ tính tay TP=80/FP=20/FN=40 → P=0,80 R=0,667 F1=0,727; trường hợp toàn FN không gây lỗi chia 0 |

Ngoài ra đã chạy kiểm thử end-to-end trên 2 frame thật trích từ video sg(3)
với nhãn giả lập (đặt trong thư mục tạm, **không** commit): xác nhận script
đọc được ảnh, phân tích được nhãn, chạy được cả hai detector và ghi ra đúng
hai file CSV với đầy đủ cột. Dữ liệu tạm đã được xoá sau khi kiểm thử.

### Báo cáo DOCX

File mới: `TransDetect-Vid_BaoCao_FINAL_v2.docx`
(bản gốc `TransDetect-Vid_BaoCao_FINAL.docx` **không bị ghi đè**).

| Mục | Thay đổi |
|---|---|
| 1.2 | Nêu rõ hai nhóm chỉ số: FPS cho hiệu năng, Precision/Recall/F1@IoU 0,5 cho chất lượng; nêu lý do gộp lớp `vehicle` |
| 4.1 | Bổ sung mô tả cách trích `[N]` frame phân bố đều, công cụ gán nhãn, và nguyên tắc không dùng tập này để chỉnh tham số |
| Bảng 4.1 | Ô ground-truth: `Chưa có` → `Có, [N] frame được gán nhãn thủ công` |
| Bảng 4.2 | Dòng "Độ đo hiện có" nay nêu cả Precision/Recall/F1, không còn nói "chưa tính được" |
| 4.2.2 | Viết lại: định nghĩa TP/FP/FN, quy tắc ghép một-một tại IoU ≥ 0,5, công thức Precision/Recall/F1, giải thích class-agnostic |
| 4.2.2 (thêm) | Đoạn mới giải thích **vì sao không dùng Accuracy** (không đếm được True Negative) và **vì sao không dùng mAP để so sánh** (nhánh truyền thống không có confidence chuẩn) |
| 4.3 | Thêm **Bảng 4.4** — bảng kết quả định lượng hai pipeline, kèm đoạn dẫn và ghi chú |
| 4.3 | Bỏ câu "cột chất lượng để chưa đo vì chưa có ground-truth", trỏ sang Bảng 4.4 |
| 5.1 | Thêm đoạn ghi nhận đã bổ sung quy trình đánh giá có ground truth, tách khỏi luồng demo nên tái lập được |
| 5.3 | "xây dựng tập kiểm thử có ground truth" → "mở rộng tập hiện có", thêm hướng class-aware/mAP **chỉ cho YOLO11** |

Bảng 4.3 (FPS) giữ nguyên nội dung. Bảng kỳ vọng định tính giữ nguyên nội
dung nhưng đổi số hiệu từ 4.4 thành 4.5 (xem mục sửa bố cục bên dưới).
Mọi hình ảnh, trang bìa, mục lục, font và bố cục không liên quan đều không
bị đụng tới.

### Placeholder chưa thể điền

Tập ground truth **chưa được gán nhãn** tại thời điểm này, nên theo nguyên tắc
không bịa số liệu, các ô sau vẫn để dạng placeholder trong DOCX:

| Placeholder | Ý nghĩa | Nguồn khi điền |
|---|---|---|
| `[N]` | Số frame được gán nhãn | Đếm file trong `evaluation/images/` |
| `[GT]` | Tổng số ground-truth box | Cột `number_of_ground_truth_boxes` |
| `[TP_C]` `[FP_C]` `[FN_C]` | TP/FP/FN của nhánh truyền thống | Dòng `Classical` trong `summary_metrics.csv` |
| `[TP_Y]` `[FP_Y]` `[FN_Y]` | TP/FP/FN của YOLO11n | Dòng `YOLO11` trong `summary_metrics.csv` |
| `[F1_C]` `[F1_Y]` | F1@0,5 của hai pipeline | Cột `f1` trong `summary_metrics.csv` |
| `[TÊN CÔNG CỤ GÁN NHÃN]` | LabelImg / CVAT / Roboflow… | Do nhóm điền |

Quy trình điền: gán nhãn → chạy `evaluate_pipelines.py` → mở
`summary_metrics.csv` → chép số thật vào Bảng 4.4 → viết nhận xét dựa trên
số thật. **Không điền số ước lượng.**

### Sửa bố cục phát hiện qua bản render

Ba lỗi chỉ lộ ra khi render thật, không thấy được khi đọc cấu trúc file:

1. **Thứ tự số hiệu bảng sai.** Bảng kết quả mới được chèn ngay sau các hình
   (trang 45), tức là *trước* bảng "Kỳ vọng định tính" (trang 46), nhưng ban
   đầu lại được đánh số 4.5 — danh mục bảng hiện ra 4.5 rồi mới tới 4.4. Đã
   đánh số lại: bảng kết quả thành **Bảng 4.4**, bảng kỳ vọng định tính thành
   **Bảng 4.5**, kèm sửa mọi tham chiếu trong thân bài.
2. **Trang trắng ở trang 50.** Nguyên nhân: một paragraph rỗng có chứa ngắt
   trang nằm ngay trước "PHỤ LỤC A" — paragraph rỗng đó chiếm trọn một trang
   rồi ngắt trang mới đẩy Phụ lục sang trang kế. Đã xoá paragraph rỗng và đặt
   `page_break_before` trực tiếp lên heading. Tài liệu giảm từ 56 xuống 55
   trang, không còn trang trắng nào.
3. **Bảng kết quả bị tách qua hai trang**, khiến dòng YOLO11n đứng lẻ ở đầu
   trang, rời khỏi dòng tiêu đề. Đã đặt `cantSplit` cho từng dòng,
   `keepNext` cho các paragraph trong bảng và cho đoạn dẫn, cùng `tblHeader`
   để dòng tiêu đề tự lặp lại nếu vẫn phải tách.

### Kiểm tra bản render

Xuất PDF bằng Word COM rồi phân tích bằng PyMuPDF:

| Hạng mục | Kết quả |
|---|---|
| Trang trắng | Không còn (đã sửa trang 50) |
| Lỗi `Error! Reference source not found` | Không có |
| Tiếng Việt có dấu | Render đúng, không lỗi font |
| Placeholder | 11/11 hiển thị đúng trong PDF |
| Thứ tự Bảng 4.1 → 4.5 | Đúng cả trong danh mục lẫn thân bài |
| Mục lục / Danh mục hình / Danh mục bảng | Đã cập nhật, có Bảng 4.4 mới |

### Còn tồn đọng

- Chưa có tập ground truth, nên Bảng 4.4 vẫn để placeholder (xem mục trên).
