# Giới hạn của tập nhãn và phạm vi đánh giá ROI

Tài liệu này ghi lại chính xác cách bộ nhãn tham chiếu được tạo ra và phạm vi mà
các con số trong `evaluation/results_yolo_roi_assisted/` và
`evaluation/results_compare_roi_assisted/` có giá trị. Đọc tài liệu này trước
khi trích bất kỳ số nào vào báo cáo hoặc slide.

## 1. Nguồn dữ liệu

100 khung hình được trích **phân bố đều** từ **một video duy nhất**
(`duong_pho_sg(3).mp4`, 1280×720, 59,94 FPS, 780 frame) ghi **một cảnh giao
thông** ở TP.HCM. Toàn bộ kết luận chỉ áp dụng cho cảnh này.

## 2. Nhãn tham chiếu KHÔNG độc lập với mô hình

Quy trình tạo nhãn thực tế:

1. Nhãn sơ bộ được sinh bằng **chính `yolo11n.pt`** tại `conf=0.10`
   (`tools/prelabel_yolo.py`).
2. Người dùng import vào CVAT, kiểm tra và chỉnh sửa.
3. Export và remap class ID về đúng thứ tự của repo
   (`tools/import_cvat_labels.py`). CVAT dùng `0=motorcycle, 1=car` trong khi
   evaluator dùng `0=car, 1=motorcycle`, nên bước remap là bắt buộc.

**Kết quả audit:** đối chiếu từng box cho thấy **3.698/3.717 box (99,5%)** giữ
nguyên toạ độ của pre-label; người dùng thêm mới 19 box và xoá 67 box.

Hệ quả trực tiếp, đo được:

- **99,6% ground-truth box có IoU ≥ 0,999** với một prediction của cùng mô hình.
- Vì mọi cặp khớp đều có IoU ≈ 1,0 nên quét ngưỡng IoU từ 0,50 đến 0,95 **không
  làm thay đổi kết quả**, dẫn tới `mAP50 = mAP50-95` — đây là **dấu hiệu của rò
  rỉ nhãn (annotation leakage)**, không phải bằng chứng mô hình định vị hoàn hảo.
- Precision gần 1,0 vì gần như mọi prediction mạnh đều đã có sẵn trong nhãn.

Do đó `annotation_independence: false` được ghi vào `run_metadata.json` của cả
hai lần chạy.

## 3. Phạm vi định lượng: ROI dọc

Chỉ những phương tiện có **tâm bounding box** nằm trong 40% phía dưới khung hình
mới tham gia tính điểm:

```text
center_y / image_height >= 0.60
```

Với ảnh 1280×720, ngưỡng tương đương `center_y >= 432 pixel`.

Lý do: nền xa của cảnh này chứa hàng chục phương tiện chỉ vài pixel, chồng lấp
dày đặc, không thể gán nhãn nhất quán. Đưa chúng vào phép đo chỉ tạo ra nhiễu
không kiểm soát được.

Quy tắc áp dụng **giống hệt nhau** cho ground truth, prediction của YOLO11 ở
**mọi confidence** (kể cả khi thu ở `conf=0.001` để dựng đường Precision-Recall),
và box ứng viên của pipeline truyền thống. Phép lọc đặt **trước** matching
TP/FP/FN. Box có tâm đúng tại biên được tính là **nằm trong** ROI.

Ảnh nguồn **không bị crop hay resize**; `imgsz=640` vẫn là kích thước sau
letterbox của YOLO, ROI chỉ áp dụng trên toạ độ ảnh gốc sau khi đã có box.

### Số lượng thực tế

| | Full-frame | Trong ROI | Ngoài ROI |
|---|---:|---:|---:|
| Tổng ground-truth box | 3.717 | **904 (24,3%)** | 2.813 |
| car | 1.116 | 54 | 1.062 |
| motorcycle | 2.258 | 850 | 1.408 |
| bus | 81 | **0** | 81 |
| truck | 262 | **0** | 262 |

## 4. Hai lớp không đo được trong ROI

**`bus` và `truck` có 0 ground-truth box trong ROI.** Xe buýt và xe tải trong
video này đều ở xa, phía trên đường biên. Vì vậy:

- `AP50` và `AP50-95` của hai lớp là `N/A` và **bị loại khỏi phép lấy trung
  bình** — `mAP` trong ROI chỉ là trung bình của `car` và `motorcycle`.
- `Precision/Recall/F1` của hai lớp hiển thị `0.0000` chỉ vì mẫu số bằng 0. Đây
  **không** phải kết quả đo được; không được diễn giải là mô hình trượt hoàn
  toàn hai lớp này.

Số mẫu của `car` trong ROI cũng chỉ có 54 box, dưới ngưỡng 10–20 đối tượng mỗi
lớp được coi là ổn định, nên kết quả lớp này cần được xem là chỉ báo sơ bộ.

## 4b. Muốn có đánh giá độc lập thì phải làm gì

Không có cách tắt. Phải **gán nhãn từ ảnh trắng**, tuyệt đối không import và
không nhìn prediction của model trong lúc gán. Quy trình đề xuất:

1. Chọn một tập con (ví dụ 30 frame) bằng quy tắc cố định, không dựa trên kết
   quả tốt/xấu của YOLO.
2. Đóng gói ZIP **không kèm nhãn** (`tools/prepare_cvat_dataset.py --labels ""`).
3. Tạo Task CVAT mới, vẽ box từ đầu trong ROI đã định nghĩa.
4. Chạy lại evaluator với `--roi-y-min 0.60` trên tập đó.

Chỉ khi ấy Precision và mAP mới có ý nghĩa độc lập.

## 5. Những điều KHÔNG được kết luận

- Không gọi `mAP` là "tỉ lệ phần trăm xe được phát hiện đúng".
- Không dùng `mAP50 = mAP50-95 = 0,995` làm bằng chứng năng lực mô hình, vì con
  số đó đến từ rò rỉ nhãn.
- Không trộn kết quả full-frame cũ với kết quả ROI. Kết quả cũ nằm ở
  `evaluation/not_for_report_results_yolo_fullframe/`, **chỉ giữ trên máy làm
  bằng chứng audit**, không commit và không dùng cho báo cáo.
- Không đặt `mAP` của YOLO11 cạnh `F1` của Classical rồi gọi là chênh lệch độ
  chính xác. Classical không có confidence chuẩn nên **không có mAP**.
- Không suy rộng sang điều kiện giao thông, thời tiết hay góc quay khác.

## 6. Chỉ số nên dùng

Ưu tiên báo cáo **Precision, Recall và F1 tại `conf=0.25`, matching IoU `0.50`,
trong ROI `y >= 0.60H`**. Đây là nhóm chỉ số ít bị rò rỉ nhãn làm sai lệch nhất,
vì chúng đo ở ngưỡng vận hành thật của demo.

Khi trình bày `mAP`, luôn kèm dấu `(*)` và chú thích ở Mục 2 của tài liệu này.

## 7. Cách tái lập

```bash
python evaluate_yolo.py --images evaluation/images --labels evaluation/labels \
  --model yolo11n.pt --imgsz 640 --conf 0.25 --predict-conf 0.001 \
  --nms-iou 0.45 --match-iou 0.50 --max-det 300 --roi-y-min 0.60 \
  --output evaluation/results_yolo_roi_assisted

python evaluate_pipelines.py --images evaluation/images --labels evaluation/labels \
  --model yolo11n.pt --conf 0.25 --iou-match 0.50 --roi-y-min 0.60 \
  --output evaluation/results_compare_roi_assisted

python tools/visualize_roi_ground_truth.py --roi-y-min 0.60
```

Commit sinh ra số liệu: `34b57bb`. Model SHA-256:
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.

Lưu ý tên tham số khác nhau giữa hai script: `evaluate_yolo.py` dùng
`--match-iou`, còn `evaluate_pipelines.py` dùng `--iou-match`.

Bỏ `--roi-y-min` sẽ cho lại đúng kết quả full-frame cũ — điều này đã được kiểm
chứng bằng regression test: bốn file CSV trùng khít từng byte với bản lưu trong
`evaluation/not_for_report_results_yolo_fullframe/`.
