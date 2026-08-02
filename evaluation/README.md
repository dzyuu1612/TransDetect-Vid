# Tập đánh giá định lượng (ground truth)

Thư mục này chứa dữ liệu để chạy **`evaluate_yolo.py`** — đánh giá định lượng
riêng cho nhánh YOLO11 bằng Precision / Recall / F1, AP@0.50 và mAP@0.50:0.95.

## Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| `images/` | **100 frame** trích từ `duong_pho_sg(3).mp4` (1280×720; 59,94 FPS; 780 frame) |
| `frame_manifest.csv` | Đã sinh — ghi frame gốc và mốc thời gian của từng ảnh |
| `labels/` | **100 file nhãn**, tổng **3.717 box** — đã qua validator |
| `results_yolo_roi_assisted/` | Kết quả chính thức (ROI `y >= 0,60H`) |
| `results_pipelines_roi_assisted/` | So sánh định vị class-agnostic, cùng ROI |

Self-test: `evaluate_yolo.py` **74/74 PASS**, `evaluate_pipelines.py` **30/30 PASS**.

### Phân bố ground truth

| | Full-frame | Trong ROI (`center_y/H >= 0,60`) | Ngoài ROI |
|---|---:|---:|---:|
| Tổng | 3.717 | **904 (24,3%)** | 2.813 |
| car | 1.116 | 54 | 1.062 |
| motorcycle | 2.258 | 850 | 1.408 |
| bus | 81 | **0** | 81 |
| truck | 262 | **0** | 262 |

### Kết quả trong ROI

| Model | GT | TP | FP | FN | Precision | Recall | F1 | mAP50 (*) | mAP50-95 (*) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 904 | 696 | 0 | 208 | 100,00% | 76,99% | **87,00%** | 99,50% | 99,50% |

| Pipeline | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Truyền thống | 8 | 665 | 896 | 1,19% | 0,88% | 1,01% |
| YOLO11 | 696 | 0 | 208 | 100,00% | 76,99% | 87,00% |

> **(*)** Nhãn ở đây **không phải ground truth độc lập**. Chúng được khởi tạo
> bằng chính `yolo11n.pt` ở `conf=0.10` rồi chỉnh trong CVAT; 99,5% box giữ
> nguyên toạ độ pre-label. Precision và mAP vì thế lạc quan. Đọc
> [`ANNOTATION_AND_ROI_LIMITATIONS.md`](ANNOTATION_AND_ROI_LIMITATIONS.md)
> trước khi trích bất kỳ số nào.

## Phạm vi đánh giá: ROI dọc

Chỉ phương tiện có **tâm bounding box** nằm trong 40% phía dưới khung hình mới
tham gia tính điểm:

```text
center_y / image_height >= 0.60      →   y >= 432px với ảnh 1280×720
```

Box có tâm **đúng tại biên** được tính là nằm trong ROI. Quy tắc áp dụng giống
hệt cho ground truth, prediction YOLO ở **mọi confidence** (kể cả khi thu ở
`conf=0.001` để dựng đường Precision-Recall) và box ứng viên của Classical. Phép
lọc đặt **trước** matching TP/FP/FN. Ảnh nguồn không bị crop hay resize.

Lý do: nền xa của cảnh này có hàng chục phương tiện chỉ vài pixel, chồng lấp dày
đặc, không gán nhãn nhất quán được.

## Cấu trúc

```text
evaluation/
├── images/               # 100 frame trích đều từ video
├── labels/               # nhãn YOLO cùng tên, ví dụ frame_000007.txt
├── classes.txt           # car, motorcycle, bus, truck
├── frame_manifest.csv    # ảnh nào ứng với frame nào, ở giây thứ mấy
└── results_yolo/         # CSV/JSON do evaluate_yolo.py sinh (không gán nhãn ở đây)
```

Mỗi ảnh phải có **đúng một** file nhãn cùng tên. Frame thật sự không có phương
tiện nào thì dùng **file rỗng** — không được để thiếu file. Script sẽ **dừng
hẳn** và không sinh file kết quả nào nếu thiếu nhãn, vì một ảnh thiếu nhãn bị
hiểu là "frame không có xe", khiến mọi box YOLO phát hiện trên ảnh đó thành
False Positive và làm Precision sai lệch âm thầm.

## Định dạng nhãn

Định dạng YOLO, mỗi dòng một phương tiện, bốn toạ độ chuẩn hoá về `[0, 1]`:

```text
class_id x_center y_center width height
```

`class_id` theo đúng thứ tự trong `classes.txt`:

```text
0 car
1 motorcycle
2 bus
3 truck
```

Khác với `evaluate_pipelines.py` (bỏ qua `class_id` vì chấm class-agnostic),
`evaluate_yolo.py` **dùng `class_id`**: một prediction chỉ được tính đúng khi
trùng cả vị trí lẫn lớp. Vì vậy `class_id` phải gán chính xác.

### Ánh xạ lớp — điểm dễ sai nhất

`yolo11n.pt` pre-train trên COCO dùng class ID nội bộ (car=2, motorcycle=3,
bus=5, truck=7), **khác hoàn toàn** với ID của tập nhãn này (car=0,
motorcycle=1, bus=2, truck=3). Nếu đem hai bộ ID so trực tiếp, mọi prediction
đều bị gán nhầm lớp — script vẫn ra số nhưng con số đó vô nghĩa.

`evaluate_yolo.py` xử lý việc này bằng cách ghép lớp qua **TÊN** đã chuẩn hoá,
không bao giờ qua ID:

```text
car             -> car
motorcycle      -> motorcycle
motorbike       -> motorcycle
bus             -> bus
truck           -> truck
container truck -> truck
```

Lớp ngoài bốn lớp trên (`person`, `bicycle`…) bị **bỏ qua**, không tính thành
False Positive — tập nhãn không gán những lớp đó nên phạt model là không công bằng.

## Quy tắc gán nhãn

1. Mỗi phương tiện nhìn thấy rõ phải có **đúng một** box.
2. Box ôm sát phần phương tiện nhìn thấy được; không lấy quá nhiều nền.
3. Xe bị che khuất **vẫn gán** nếu còn đủ phần nhận biết được.
4. Xe nhỏ ở xa và xe ở mép ảnh vẫn gán nếu còn nhận diện được.
5. **Không gán** biển báo, người đi bộ, xe đạp — báo cáo chỉ có bốn lớp trên.
6. **Không dùng output của YOLO làm ground truth.** Lấy dự đoán của YOLO làm
   chuẩn rồi chấm điểm chính YOLO là vòng lặp tự khẳng định, kết quả luôn ra
   gần 1,00 và hoàn toàn vô nghĩa.
7. Được phép dùng YOLO để **pre-label** cho nhanh, nhưng người gán nhãn phải
   xem và sửa **toàn bộ**: box sai vị trí, sai lớp, box thiếu và box thừa.
8. **Không bỏ qua một chiếc xe** chỉ vì model không phát hiện được nó.

## Quy trình

### Bước 1 — Trích frame (ĐÃ XONG)

```bash
python prepare_evaluation_frames.py \
  --video "duong_dan/duong_pho_sg(3).mp4" \
  --count 100 \
  --output evaluation/images \
  --manifest evaluation/frame_manifest.csv
```

Frame được lấy **phân bố đều** trên toàn video bằng `np.linspace`, không lấy 100
frame liên tiếp: ở 59,94 FPS các frame liền nhau gần như giống hệt nhau (xe chỉ
dịch 1–2 pixel), lấy liên tiếp thì thực chất chỉ đánh giá trên ~1,7 giây video.

### Bước 2 — Gán nhãn (VIỆC PHẢI LÀM)

Ba công cụ hỗ trợ trong `tools/` giúp rút ngắn công việc này. Chúng **chỉ chuẩn
bị dữ liệu**, không tạo ra ground truth.

```bash
# 2a. Sinh nhãn sơ bộ bằng YOLO (conf thấp 0,10 cho đỡ bỏ sót)
python tools/prelabel_yolo.py

# 2b. Đóng gói ảnh + nhãn sơ bộ thành ZIP để import vào CVAT
python tools/prepare_cvat_dataset.py --labels evaluation/prelabels

# 2c. Xem nhãn bằng mắt bất cứ lúc nào
python tools/visualize_ground_truth.py --labels evaluation/prelabels
```

Nhãn sơ bộ được ghi vào `evaluation/prelabels/`, **không bao giờ** ghi thẳng vào
`evaluation/labels/`. Trong CVAT phải mở **từng ảnh** và sửa bốn loại lỗi:

1. **Box thiếu** — quan trọng nhất. Xe nhỏ ở xa, xe đỗ, xe ở mép ảnh. Bỏ sót
   ground truth sẽ khiến prediction ĐÚNG của YOLO bị tính nhầm thành False
   Positive, làm Precision sai.
2. **Box thừa** — vùng không phải xe, hoặc một xe bị tách thành hai box.
3. **Sai lớp** — hay gặp nhất là xe khách bị gán thành `truck`.
4. **Box không ôm sát** phần phương tiện nhìn thấy được.

Chỉ sau khi kiểm tra bằng mắt toàn bộ ảnh mới chép nhãn sang
`evaluation/labels/`. Ghi lại **công cụ, người gán nhãn, người kiểm tra chéo và
ngày hoàn tất** để đưa vào Mục 4.1 của báo cáo.

Khuyến nghị kiểm tra chéo: một người gán toàn bộ, người thứ hai kiểm tra ít nhất
10–20% số frame. Nếu nhóm có bốn thành viên, chia ~25 frame/người và kiểm tra
chéo tối thiểu 5 frame/người.

> `evaluate_yolo.py --validate-only` chỉ kiểm tra **định dạng** — đủ file,
> `class_id` trong `0..3`, toạ độ trong `[0,1]`. Nó hoàn toàn không biết box có
> được vẽ đúng lên chiếc xe hay không. Vì vậy phải dùng
> `tools/visualize_ground_truth.py` và nhìn bằng mắt.

### Bước 3 — Kiểm tra nhãn

```bash
python evaluate_yolo.py --validate-only
```

Lệnh này kiểm tra: đủ file nhãn, không có nhãn thừa, `class_id` trong `0..3`,
toạ độ nằm trong `[0, 1]`, width/height dương, box không suy biến sau khi đổi
sang pixel. Phải sửa hết lỗi trước khi sang bước sau.

### Bước 4 — Chạy đánh giá

Lệnh đã thực sự chạy để sinh số liệu ở đầu tài liệu:

```bash
python evaluate_yolo.py \
  --images evaluation/images \
  --labels evaluation/labels \
  --model yolo11n.pt \
  --imgsz 640 --conf 0.25 --predict-conf 0.001 \
  --nms-iou 0.45 --match-iou 0.50 --max-det 300 --roi-y-min 0.60 \
  --output evaluation/results_yolo_roi_assisted

python evaluate_pipelines.py \
  --images evaluation/images --labels evaluation/labels \
  --model yolo11n.pt --conf 0.25 --iou-match 0.50 --roi-y-min 0.60 \
  --output evaluation/results_pipelines_roi_assisted

python tools/visualize_roi_ground_truth.py --roi-y-min 0.60
```

Nếu bạn có sẵn frame trên máy và muốn tái lập: đặt ảnh vào `evaluation/images/`,
nhãn vào `evaluation/labels/`, chạy `--validate-only` rồi chạy đúng ba lệnh trên.
Bỏ `--roi-y-min` sẽ cho lại kết quả full-frame.

Sinh ra năm file:

| File | Nội dung |
|---|---|
| `summary_metrics.csv` | Một dòng tổng: TP/FP/FN, P/R/F1, mAP50, mAP50-95 |
| `per_class_metrics.csv` | Bốn lớp: GT, TP/FP/FN, P/R/F1, AP50, AP50-95 |
| `per_frame_metrics.csv` | Từng ảnh: GT, số prediction, TP/FP/FN |
| `predictions.csv` | Mọi prediction kèm confidence và toạ độ |
| `run_metadata.json` | Commit, SHA256 model, mọi ngưỡng, phiên bản thư viện |

`predictions.csv` cho phép kiểm tra lại một con số bất thường mà không phải
chạy lại model.

## Hai ngưỡng IoU khác nhau — không được nhầm

| Ngưỡng | Giá trị | Vai trò |
|---|---|---|
| NMS IoU | 0,45 | Ultralytics dùng để loại prediction trùng nhau **trong lúc suy luận** |
| Matching IoU | 0,50 (và 0,50→0,95 cho mAP) | Evaluator dùng để quyết định prediction có **khớp ground truth** hay không |

Hai giá trị có vai trò hoàn toàn khác nhau, không gộp thành một biến.

## Vì sao chạy suy luận ở `conf=0.001`

Precision/Recall/F1 được báo cáo tại `conf=0.25` (đúng cấu hình demo), nhưng AP
cần cả những prediction điểm thấp ở phần đuôi để dựng trọn đường
Precision-Recall. Nếu chỉ chạy ở 0,25 thì đường PR bị cắt cụt và AP thấp hơn giá
trị thật. Vì vậy script chạy model **một lượt duy nhất** ở `conf=0.001` rồi lọc
lại ở 0,25 cho nhóm chỉ số P/R/F1.

## Giới hạn phải ghi rõ trong báo cáo

Cả 100 frame đến từ **một video 13 giây, một cảnh giao thông**. Vì vậy đây là
**tập đánh giá nội bộ trên một cảnh**, không phải bằng chứng về khả năng tổng
quát cho mọi video giao thông. Nếu một lớp có ít hơn 10–20 đối tượng thì vẫn báo
số thật nhưng phải ghi rõ kết quả của lớp đó chưa ổn định.

Nếu sau này fine-tune model, phải chia train/validation/test **theo video
nguồn**, không chia ngẫu nhiên các frame liền nhau vì sẽ gây rò rỉ thời gian.

## Lưu ý về dung lượng

`.gitignore` loại trừ `*.jpg`, `*.png` nên ảnh trong `images/` **không** được
commit. Nếu cần nộp kèm bằng chứng, commit riêng 20–30 frame minh chứng bằng
`git add -f`, còn toàn bộ tập đánh giá lưu ở Google Drive/Roboflow và ghi URL
cùng phiên bản dataset vào đây.
