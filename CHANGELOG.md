# CHANGELOG

Ghi lại các thay đổi theo từng đợt làm việc. Mục đích là để đối chiếu ngược
giữa mã nguồn và báo cáo: mỗi con số trong báo cáo phải truy được về đúng
commit, đúng lệnh chạy và đúng file kết quả.

---

## [Chưa phát hành] — Bổ sung quy trình đánh giá định lượng

Đặc tả nguồn: `BO_SUNG_DANH_GIA_DO_CHINH_XAC_TRANDETECT_VID.md`
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

`python evaluate_pipelines.py --selftest` — **20/20 PASS**:

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
| 4.3 | Thêm **Bảng 4.5** — bảng kết quả định lượng hai pipeline, kèm đoạn dẫn và ghi chú |
| 4.3 | Bỏ câu "cột chất lượng để chưa đo vì chưa có ground-truth", trỏ sang Bảng 4.5 |
| 5.1 | Thêm đoạn ghi nhận đã bổ sung quy trình đánh giá có ground truth, tách khỏi luồng demo nên tái lập được |
| 5.3 | "xây dựng tập kiểm thử có ground truth" → "mở rộng tập hiện có", thêm hướng class-aware/mAP **chỉ cho YOLO11** |

Bảng 4.3 (FPS) và Bảng 4.4 (kỳ vọng định tính) giữ nguyên. Mọi hình ảnh,
trang bìa, mục lục, font và bố cục không liên quan đều không bị đụng tới.

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
`summary_metrics.csv` → chép số thật vào Bảng 4.5 → viết nhận xét dựa trên
số thật. **Không điền số ước lượng.**

### Còn tồn đọng

- Số trang trong Mục lục / Danh mục hình / Danh mục bảng có thể lệch sau khi
  thêm Bảng 4.5. Cách sửa: mở file trong Word, `Ctrl+A` → `F9` → chọn
  "Update entire table".
- Chưa render toàn bộ DOCX ra PDF để kiểm tra trực quan (tràn bảng, trang
  trắng, lỗi font). Máy hiện không có LibreOffice, còn tự động hoá Word bị
  chậm/treo nhiều lần trong phiên làm việc này.
