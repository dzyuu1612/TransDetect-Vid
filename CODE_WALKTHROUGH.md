# TransDetect-Vid — Giải thích mã nguồn từ đầu tới cuối

Tài liệu đi kèm `TransDetect-Vid_BaoCao_v14.docx`.
Mục tiêu: đọc xong biết **dòng nào nằm ở đâu**, **làm gì**, và **chạy theo thứ tự nào**.

Cách đọc: tài liệu sắp theo **thứ tự thực thi thật**, không sắp theo thứ tự file.
Chọn một trong hai kịch bản ở Phần 0 rồi đi thẳng theo nó.

---

## Mục lục

- [Phần 0 — Bản đồ repo và hai kịch bản chạy](#phần-0--bản-đồ-repo-và-hai-kịch-bản-chạy)
- [Phần 1 — Kịch bản A: chạy CLI (`main.py`)](#phần-1--kịch-bản-a-chạy-cli-mainpy)
- [Phần 2 — Các module lõi](#phần-2--các-module-lõi)
  - [2.1 `config.py`](#21-configpy--bảng-tham-số)
  - [2.2 `preprocessing.py`](#22-preprocessingpy--tiền-xử-lý-listing-31)
  - [2.3 `classical_detector.py`](#23-classical_detectorpy--phát-hiện-truyền-thống-listing-32)
  - [2.4 `optical_flow.py`](#24-optical_flowpy--lucaskanade-listing-33)
  - [2.5 `yolo_detector.py`](#25-yolo_detectorpy--nhánh-học-sâu-listing-34)
  - [2.6 `visualization.py`](#26-visualizationpy--vẽ-kết-quả)
  - [2.7 `__init__.py`](#27-__init__py--khai-báo-package)
- [Phần 3 — Kịch bản B: Dashboard (`app_streamlit.py`)](#phần-3--kịch-bản-b-dashboard-app_streamlitpy)
- [Phần 3.8 — Sửa lỗi tương phản chữ (theme + màu CSS)](#phần-38--sửa-lỗi-tương-phản-chữ-theme--màu-css)
- [Phần 4 — `legacy/` và các script cũ](#phần-4--legacy-và-các-script-cũ)
- [Phần 5 — Sơ đồ tổng](#phần-5--sơ-đồ-tổng)

---

## Phần 0 — Bản đồ repo và hai kịch bản chạy

Repo chứa **ba thế hệ** của cùng một thuật toán. Phân biệt được ba lớp này là
điều dễ nhầm nhất:

| Lớp | File | Trạng thái |
|---|---|---|
| **Hiện hành (Chương 3 báo cáo)** | `src/transdetect/*.py` | Chuẩn. Báo cáo trích ở Listing 3.1–3.4. |
| **Legacy (bản nháp học)** | `legacy/*.py` | Đông lạnh. **Khác thuật toán**. Không ai import. |
| **Script cũ ở gốc** | `yolo_detector.py`, `app.py`, `run_demo.py`, `train_yolo.py` | Prototype trước refactor. |

Hai kịch bản chạy:

```
KỊCH BẢN A — CLI
  python main.py --input v.mp4 --method classical --output out.mp4
    main.py  →  pipeline.py  →  {preprocessing, classical_detector,
                                 optical_flow, yolo_detector, visualization}
    Kết quả: một file .mp4 đã vẽ box (+ mũi tên nếu classical).

KỊCH BẢN B — Dashboard
  streamlit run app_streamlit.py
    app_streamlit.py  →  {config, preprocessing, classical_detector,
                          optical_flow, yolo_detector, visualization}
    KHÔNG đi qua pipeline.py. Kết quả: hiển thị trực tiếp trên trình duyệt
    + xuất CSV/JSON. Đây là nguồn của Hình 4.1–4.4.
```

Điểm mấu chốt: **Dashboard không dùng `pipeline.py`.** Nó gọi thẳng các module
lõi và tự viết lại vòng lặp frame của riêng nó. Hai vòng lặp này phải được giữ
đồng bộ bằng tay — đó là lý do trước đây Dashboard thiếu Lucas-Kanade.

---

## Phần 1 — Kịch bản A: chạy CLI (`main.py`)

### 1.1 `main.py` — 27 dòng, chỉ làm mỗi việc phân tích tham số

```python
import argparse                          # dòng 5  — thư viện chuẩn, đọc tham số dòng lệnh
from src.transdetect import pipeline      # dòng 6  — toàn bộ logic nằm bên kia, file này không xử lý gì
```

```python
def main():                                                        # dòng 9
    p = argparse.ArgumentParser(description="...")                 # dòng 10 — tạo bộ đọc tham số
    p.add_argument("--input",  required=True)                      # dòng 11 — thiếu → argparse tự báo lỗi và thoát
    p.add_argument("--output", required=True)                      # dòng 12
    p.add_argument("--method", choices=["classical","yolo"], required=True)  # dòng 13
    p.add_argument("--model",   default=None)                      # dòng 14
    p.add_argument("--conf",    type=float, default=None)          # dòng 15
    p.add_argument("--iou",     type=float, default=None)          # dòng 16
    p.add_argument("--max-det", dest="max_det", type=int, default=None)  # dòng 17
    a = p.parse_args()                                             # dòng 18 — đọc sys.argv, trả về object
```

Ba chi tiết đáng học ở đây:

- `choices=[...]` (dòng 13): argparse tự chặn giá trị sai, ta không cần viết
  `if` kiểm tra. Gõ `--method yolo11` sẽ bị từ chối ngay.
- `dest="max_det"` (dòng 17): CLI dùng gạch nối `--max-det`, nhưng Python không
  cho tên biến có gạch nối. `dest=` đổi tên thành `a.max_det`.
- `default=None` **chứ không phải một con số** (dòng 14–17): đây là mấu chốt.
  `None` nghĩa là "người dùng không truyền". `pipeline.py` sẽ thấy `None` và lấy
  giá trị từ `config.py`. Nhờ vậy chỉ có **một** nơi định nghĩa mặc định. Nếu
  đặt `default=0.25` ở đây thì sẽ có hai nơi, và sửa `config.py` sẽ không có tác dụng.

```python
    if a.method == "classical":                                    # dòng 19
        pipeline.run_classical(a.input, a.output)                  # dòng 20 — nhánh classical KHÔNG nhận conf/iou
    else:                                                          # dòng 21
        pipeline.run_yolo(a.input, a.output, model_path=a.model,   # dòng 22
                          conf=a.conf, iou=a.iou, max_det=a.max_det)
```
Truyền `--conf 0.5` cùng `--method classical` sẽ bị **bỏ qua im lặng** — nhánh
truyền thống không có khái niệm confidence.

```python
if __name__ == "__main__":     # dòng 26 — chỉ chạy khi gọi trực tiếp `python main.py`,
    main()                     # dòng 27   không chạy khi file bị import từ nơi khác
```

### 1.2 `src/transdetect/pipeline.py` — vòng lặp video của CLI

#### Imports (dòng 12–20)

```python
import os                                              # dòng 12 — dùng cho makedirs/dirname
import cv2                                             # dòng 14
from . import config                                   # dòng 16 — dấu chấm = import tương đối
from . import preprocessing                            # dòng 17   trong cùng package transdetect
from . import classical_detector                       # dòng 18
from . import visualization                            # dòng 19
from .optical_flow import LucasKanadeTracker           # dòng 20
from .yolo_detector import Yolo11VehicleDetector       # dòng 21
```
`from . import X` khác `import X`: dấu chấm buộc Python tìm trong chính package
`src/transdetect/`, không tìm ngoài site-packages. Nhờ đó `config.py` của dự án
không bị nhầm với một thư viện tên `config` nào khác.

#### `_open_capture()` — dòng 24–34

```python
def _open_capture(input_path):                                     # dòng 24
    cap = cv2.VideoCapture(input_path)                             # dòng 31
    if not cap.isOpened():                                         # dòng 32
        raise FileNotFoundError(f"Không mở được video đầu vào: {input_path}")  # dòng 33
    return cap                                                     # dòng 34
```
**Vì sao dòng 32 là bắt buộc:** `cv2.VideoCapture` với đường dẫn sai **không ném
lỗi** — nó trả về một đối tượng bình thường nhưng ở trạng thái "đóng". Frame đầu
`cap.read()` trả `False`, vòng `while` thoát ngay, chương trình in ra "Đã lưu
video kết quả" trong khi file rỗng. Hỏng mà không có dấu hiệu nào.

Dấu `_` đầu tên hàm là quy ước Python: "hàm nội bộ, đừng gọi từ ngoài module".

#### `_open_writer()` — dòng 37–59

```python
def _open_writer(cap, output_path):                                # dòng 37
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25                       # dòng 38
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                # dòng 39
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))               # dòng 40
```
`or 25` (dòng 38): `cap.get()` trả `float`. Video có metadata hỏng trả `0.0`,
mà `0.0` là falsy trong Python nên `or` sẽ nhảy sang `25`. Không có dòng này,
`VideoWriter` nhận fps=0 và tạo file không phát được.

`int(...)` (dòng 39–40): `cap.get()` luôn trả float (`1920.0`), nhưng
`VideoWriter` cần tuple số nguyên.

```python
    if width <= 0 or height <= 0:                                  # dòng 44
        cap.release()                                              # dòng 45
        raise ValueError(f"...: {width}x{height}")                 # dòng 46-48
```
`cap.release()` **trước** khi `raise` (dòng 45): nếu ném lỗi mà không release,
handle file video bị treo cho đến khi Python thu gom rác — trên Windows điều đó
khoá file, không xoá/ghi đè được.

```python
    output_dir = os.path.dirname(os.path.abspath(output_path))     # dòng 51
    os.makedirs(output_dir, exist_ok=True)                         # dòng 52
```
`os.path.abspath` trước `dirname` (dòng 51): nếu người dùng gõ `--output out.mp4`
(không có thư mục), `dirname("out.mp4")` trả chuỗi rỗng `""` và `makedirs("")`
sẽ lỗi. `abspath` biến nó thành `D:\CV\out.mp4` nên `dirname` trả `D:\CV`.

`exist_ok=True` (dòng 52): chạy lần thứ hai không báo lỗi "thư mục đã tồn tại".
Đây là lý do bạn **không cần tự tạo** `outputs/`.

```python
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")                       # dòng 54
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # dòng 55
    if not writer.isOpened():                                      # dòng 56
        cap.release()                                              # dòng 57
        raise IOError(f"Không khởi tạo được VideoWriter cho: {output_path}")  # dòng 58
    return writer                                                  # dòng 59
```
`*"mp4v"` (dòng 54): dấu `*` bung chuỗi thành 4 ký tự rời `'m','p','4','v'`, vì
`VideoWriter_fourcc` nhận 4 tham số chứ không nhận một chuỗi.

Ba lần kiểm tra ở trên chặn **ba kiểu thất bại im lặng**: kích thước sai → mọi
`write()` không làm gì; thư mục không tồn tại → writer hỏng lặng lẽ; codec `mp4v`
không có trên máy → `isOpened()` trả `False`.

#### `run_classical()` — dòng 62–95

```python
def run_classical(input_path, output_path):                        # dòng 62
    cap       = _open_capture(input_path)                          # dòng 64 — đã kiểm tra, chắc chắn đọc được
    writer    = _open_writer(cap, output_path)                     # dòng 65
    tracker   = LucasKanadeTracker()                               # dòng 66 — MỘT instance cho cả video
    prev_gray = None                                               # dòng 67 — cờ canh: frame 1 chưa có frame trước
```
Dòng 66 nằm **ngoài** vòng lặp là bắt buộc: `LucasKanadeTracker` giữ
`self.prev_points` xuyên frame. Tạo lại mỗi frame thì mọi điểm bám sẽ mất sạch,
và bạn chỉ nhận được các mũi tên rời rạc vô nghĩa.

```python
    while True:                                                    # dòng 69
        ret, frame = cap.read()                                    # dòng 70
        if not ret:                                                # dòng 71
            break                                                  # dòng 72 — hết video hoặc lỗi giải mã
```
`cap.read()` trả về **tuple 2 phần tử**: `ret` (bool, đọc được hay không) và
`frame` (mảng NumPy shape `(H, W, 3)`, kiểu `uint8`, thứ tự kênh **BGR**).

```python
        pre = preprocessing.preprocess_frame(frame, config.MEDIAN_KERNEL_SIZE)  # dòng 74
        boxes, _, _ = classical_detector.detect_vehicle_candidates(             # dòng 75
            pre,                                                   # dòng 76
            min_area=config.MIN_CONTOUR_AREA,                      # dòng 77
            max_area=config.MAX_CONTOUR_AREA,                      # dòng 78
            max_aspect_ratio=config.MAX_ASPECT_RATIO,              # dòng 79
            edge_threshold=config.SOBEL_EDGE_THRESHOLD,            # dòng 80
        )
        output = visualization.draw_classical_boxes(frame, boxes)  # dòng 82 — vẽ lên frame MÀU
```
`boxes, _, _` (dòng 75): hàm trả về 3 giá trị `(boxes, combined_mask, threshold)`.
Dấu `_` là quy ước "tôi biết có giá trị này nhưng không dùng". Hai giá trị bỏ đi
là mask kết hợp và ngưỡng T — chúng chỉ hữu ích khi debug.

Dòng 82 truyền `frame` (ảnh **màu**) chứ không phải `pre` (ảnh xám), vì video
kết quả phải giữ màu gốc; `pre` chỉ dùng để *tính toán*.

```python
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # dòng 84 — ảnh xám THÔ
        if prev_gray is not None:                                  # dòng 85
            points, motion = tracker.track_features(prev_gray, curr_gray)  # dòng 86
            if len(points) > 0:                                    # dòng 87
                output = visualization.draw_motion_vectors(output, points, motion)  # dòng 88
        prev_gray = curr_gray                                      # dòng 89 — đẩy cửa sổ tiến 1 bước
        writer.write(output)                                       # dòng 91
```
**Điểm quan trọng nhất của cả file:** dòng 84 tạo ảnh xám **thô**, không dùng
`pre`. Lý do: `equalizeHist` bên trong `preprocess_frame` đổi phép ánh xạ cường
độ ở *từng* frame, nên cùng một điểm trên xe sẽ có giá trị xám khác nhau giữa
hai frame liên tiếp → vi phạm giả định độ sáng không đổi của Lucas-Kanade
(§2.5.1) → tracker bám sai. Vì vậy **phát hiện và theo vết cố ý dùng hai ảnh xám
khác nhau**.

Dòng 88 vẽ **đè** lên `output` (đã có box), không vẽ lên `frame`. Thứ tự: box
trước, mũi tên sau.

Frame đầu tiên: `prev_gray is None` → chỉ có box, không có mũi tên.

```python
    cap.release()                                                  # dòng 93
    writer.release()                                               # dòng 94 — BẮT BUỘC: không release thì
    print(f"Đã lưu video kết quả tại: {output_path}")              # dòng 95   file mp4 thiếu phần header cuối
```
`writer.release()` ghi nốt phần đuôi file (moov atom). Quên dòng này thì file
.mp4 tồn tại nhưng không trình phát nào mở được.

#### `run_yolo()` — dòng 98–120

```python
def run_yolo(input_path, output_path, model_path=None, conf=None, iou=None,   # dòng 98
             max_det=None):                                        # dòng 99
    detector = Yolo11VehicleDetector(model_path or config.DEFAULT_YOLO_MODEL)  # dòng 101
    conf    = config.CONF_THRESHOLD if conf    is None else conf    # dòng 102
    iou     = config.IOU_THRESHOLD  if iou     is None else iou     # dòng 103
    max_det = config.MAX_DET        if max_det is None else max_det # dòng 104
```
So sánh dòng 101 với 102–104 — **hai kỹ thuật khác nhau, có lý do**:

- Dòng 101 dùng `or`: `model_path` là chuỗi. Chuỗi rỗng `""` cũng không phải
  đường dẫn hợp lệ, nên gộp `None` và `""` vào một nhánh là đúng.
- Dòng 102–104 dùng `is None`: `conf` là số. Nếu viết `conf or 0.25` thì
  `--conf 0.0` (giá trị hợp lệ, nghĩa là "giữ mọi box") sẽ **âm thầm** biến
  thành `0.25`, vì `0.0` là falsy. Đây là loại bug rất khó tìm.

```python
    cap    = _open_capture(input_path)                             # dòng 106
    writer = _open_writer(cap, output_path)                        # dòng 107
    while True:                                                    # dòng 109
        ret, frame = cap.read()                                    # dòng 110
        if not ret: break                                          # dòng 111-112
        detections = detector.detect_frame(frame, conf, iou, max_det)  # dòng 114
        output     = visualization.draw_yolo_detections(frame, detections)  # dòng 115
        writer.write(output)                                       # dòng 116
    cap.release(); writer.release()                                # dòng 118-119
```
Dòng 101 nằm ngoài vòng lặp: nạp model rất tốn kém (đọc trọng số + copy sang
GPU). Tạo lại mỗi frame sẽ khiến FPS tụt xuống gần 0.

Nhánh YOLO **không có** `prev_gray`/`tracker` — nó xử lý từng frame độc lập,
không cần biết frame trước.

---

## Phần 2 — Các module lõi

### 2.1 `config.py` — bảng tham số

Không có logic, chỉ có hằng số. Mọi module khác đọc từ đây.

```python
MEDIAN_KERNEL_SIZE   = 5        # dòng 11 — k của cửa sổ median k×k (§2.3.2). Bắt buộc lẻ.
MIN_CONTOUR_AREA     = 500      # dòng 14 — contour < 500 px² = nhiễu
MAX_CONTOUR_AREA     = 150000   # dòng 15 — contour > mức này = mảng nền (trời, mặt đường)
MAX_ASPECT_RATIO     = 4.0      # dòng 16 — chặn TRÊN của w/h. Chặn dưới 1/4 được SUY RA, không lưu riêng.
SOBEL_EDGE_THRESHOLD = 40       # dòng 17 — gradient vượt mức này thì pixel là biên
```

```python
DEFAULT_YOLO_MODEL = "yolo11n.pt"   # dòng 20 — bản nano, Ultralytics tự tải lần đầu
CONF_THRESHOLD     = 0.25           # dòng 21 — loại box dưới 25% TRƯỚC NMS
IOU_THRESHOLD      = 0.45           # dòng 22 — NMS loại box chồng lấp >45% với box tốt hơn
MAX_DET            = 300            # dòng 23 — trần số box mỗi frame sau NMS
```

```python
CLASS_ALIAS = {                     # dòng 28-35
    "car": "Car", "motorcycle": "Motorcycle", "motorbike": "Motorcycle",
    "bus": "Bus", "truck": "Truck", "container truck": "Truck",
}
DISPLAY_CLASSES = ("Car", "Motorcycle", "Bus", "Truck")   # dòng 36
```
**Vì sao cần bảng này (§3.5.3):** bộ phát hiện nhận 6 tên lớp thô (để model COCO
*lẫn* model tự huấn luyện đều chạy được), nhưng UI chỉ có 4 ô đếm. Dict này gộp
6 → 4. Tên nào **không có** trong bảng bị loại, không bao giờ cộng nhầm sang lớp
bên cạnh.

`DISPLAY_CLASSES` là `tuple` chứ không phải `list` — tuple bất biến, tránh việc
một module lỡ tay `.append()` vào nó làm hỏng thứ tự 4 ô trên UI.

```python
BOX_COLOR     = (0, 255, 0)     # dòng 39 — BGR chứ KHÔNG phải RGB. Đây là xanh lá.
BOX_THICKNESS = 2               # dòng 40
```
OpenCV dùng thứ tự **BGR**. Nếu tưởng là RGB thì `(0,255,0)` bạn nghĩ là xanh lá
— may là trùng, nhưng `(255,0,0)` sẽ ra xanh dương chứ không phải đỏ.

### 2.2 `preprocessing.py` — tiền xử lý (Listing 3.1)

```python
def histogram_equalization(gray_img):                              # dòng 15
    if gray_img.ndim != 2:                                         # dòng 17
        raise ValueError("gray_img phải là ảnh xám một kênh.")     # dòng 18
    return cv2.equalizeHist(gray_img)                              # dòng 19
```
`ndim` là **số chiều** của mảng NumPy. Ảnh xám là `(H, W)` → `ndim == 2`. Ảnh
màu là `(H, W, 3)` → `ndim == 3`. `cv2.equalizeHist` chỉ nhận ảnh 1 kênh, đưa
ảnh màu vào sẽ lỗi khó hiểu ở tầng C++ — dòng 17 chặn sớm với thông báo rõ ràng.

`cv2.equalizeHist` hiện thực đúng phép ánh xạ CDF ở §2.3.1: đếm số pixel theo
từng mức xám, dựng tổng tích lũy, rồi ánh xạ mức `i` → `round((L-1)·cum(i)/N)`.
Các dải mức xám dày đặc bị kéo giãn → tương phản tăng → bước ngưỡng phía sau có
"thung lũng" rõ hơn để đặt ngưỡng.

```python
def median_filter(gray_img, kernel_size=5):                        # dòng 22
    if kernel_size < 3 or kernel_size % 2 == 0:                    # dòng 24
        raise ValueError("kernel_size phải là số lẻ và không nhỏ hơn 3.")  # dòng 25
    return cv2.medianBlur(gray_img, kernel_size)                   # dòng 26
```
Hai điều kiện ở dòng 24: **lẻ** để cửa sổ có pixel tâm thật sự (4×4 thì tâm nằm
đâu?), và **≥3** để thực sự phủ được lân cận (kernel 1 thì không làm gì).

Cách `medianBlur` chạy: với mỗi pixel, lấy 5×5 = 25 giá trị lân cận, **sắp xếp**,
lấy phần tử thứ 13 (trung vị), ghi vào tâm. Nhiễu muối tiêu là giá trị cực trị →
sắp xếp xong nó nằm ở hai đầu → không bao giờ được chọn. Biên được giữ vì trung
vị không lấy trung bình vắt qua biên (khác lọc trung bình).

```python
def preprocess_frame(bgr_frame, kernel_size=5):                    # dòng 29
    if bgr_frame is None or bgr_frame.size == 0:                   # dòng 31
        raise ValueError("Khung hình đầu vào rỗng.")               # dòng 32
    gray      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)        # dòng 34
    equalized = histogram_equalization(gray)                       # dòng 35
    denoised  = median_filter(equalized, kernel_size)              # dòng 36
    return denoised                                                # dòng 37
```
Dòng 31 kiểm tra **hai** thứ: `is None` (không đọc được frame) và `.size == 0`
(mảng tồn tại nhưng rỗng). Phải kiểm `is None` trước, vì `None.size` sẽ lỗi.

Dòng 34 áp công thức §2.3: `Gray = 0.114·B + 0.587·G + 0.299·R`. Trọng số không
đều vì mắt người nhạy với xanh lá nhất, xanh dương ít nhất.

**Thứ tự dòng 35 → 36 rất quan trọng:** cân bằng histogram *rồi mới* khử nhiễu.
Cân bằng khuếch đại nhiễu, nên median phải đứng sau để dọn phần nhiễu vừa bị
khuếch đại. Đảo ngược thì cân bằng sẽ khuếch đại lại đúng cái vừa khử.

### 2.3 `classical_detector.py` — phát hiện truyền thống (Listing 3.2)

#### a) Ngưỡng toàn cục lặp — dòng 15–39

```python
def iterative_global_threshold(gray_img, epsilon=1e-3, max_iter=100):   # dòng 15
    if gray_img.ndim != 2: raise ValueError(...)                        # dòng 17-18
    threshold = float(np.mean(gray_img))                                # dòng 20 — T₀ = mức xám trung bình
    for _ in range(max_iter):                                           # dòng 21 — chốt an toàn chống lặp vô hạn
        foreground = gray_img[gray_img >  threshold]                    # dòng 22
        background = gray_img[gray_img <= threshold]                    # dòng 23
```
Dòng 22–23 là **boolean mask indexing** của NumPy: `gray_img > threshold` tạo ra
một mảng `True/False` cùng kích thước ảnh, rồi dùng nó lập chỉ mục sẽ trả về
mảng **phẳng 1 chiều** chỉ chứa các pixel thoả điều kiện. Nhanh hơn vòng `for`
thủ công hàng trăm lần.

```python
        if foreground.size == 0 or background.size == 0:                # dòng 26
            break                                                       # dòng 27
```
Ảnh toàn đen hoặc toàn trắng → một nhóm rỗng → `np.mean([])` trả `NaN` kèm
cảnh báo. Dòng 26 thoát trước khi điều đó xảy ra.

```python
        mean_foreground = float(np.mean(foreground))                    # dòng 29
        mean_background = float(np.mean(background))                    # dòng 30
        new_threshold   = (mean_foreground + mean_background) / 2.0     # dòng 31 — T_mới = (m₁+m₂)/2
        if abs(new_threshold - threshold) < epsilon:                    # dòng 33
            threshold = new_threshold                                   # dòng 34
            break                                                       # dòng 35 — đã hội tụ
        threshold = new_threshold                                       # dòng 36 — chưa hội tụ, lặp tiếp
    binary_mask = np.where(gray_img > threshold, 255, 0).astype(np.uint8)  # dòng 38
    return binary_mask, threshold                                       # dòng 39
```
Bản chất: **phân cụm 2-means trên cường độ 1 chiều**. Mỗi vòng lặp chia lại tại
trung điểm hai giá trị trung bình cụm. Luôn hội tụ (dãy đơn điệu bị chặn), thường
dưới 10 vòng — `max_iter=100` thực tế không bao giờ chạm tới.

Dòng 38: `np.where(điều_kiện, giá_trị_nếu_đúng, giá_trị_nếu_sai)` — vectorised
if-else. `.astype(np.uint8)` bắt buộc vì `np.where` trả về `int64`, mà OpenCV chỉ
nhận `uint8` cho ảnh nhị phân.

`epsilon=1e-3` chặt hơn nhiều bản legacy (`0.5`); vì mask được ngưỡng hoá trên
`threshold` kiểu **float** (không phải int), epsilon chặt chỉ tốn thêm vài vòng
nhưng đổi lại độ ổn định dưới mức một đơn vị xám.

#### b) Độ lớn gradient Sobel — dòng 42–51

```python
def sobel_edge_detection(gray_img, edge_threshold=50):                  # dòng 42
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)             # dòng 44 — dx=1,dy=0 → biên DỌC
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)             # dòng 45 — dx=0,dy=1 → biên NGANG
```
`cv2.CV_64F` (float64) là **bắt buộc**. Đạo hàm có dấu, mà ảnh vào là `uint8`
(0–255, không dấu). Nếu yêu cầu đầu ra `uint8` thì mọi gradient âm bị cắt về 0
và bạn **mất một nửa số biên** — cụ thể là một phía của mọi vật thể.

`ksize=3` chính là hai kernel 3×3 Kx, Ky trong §2.4.2.

```python
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)                      # dòng 46 — chuẩn L2 ‖∇I‖
    magnitude = np.uint8(np.clip(magnitude, 0, 255))                    # dòng 47
    _, edge_mask = cv2.threshold(magnitude, edge_threshold, 255, cv2.THRESH_BINARY)  # dòng 48-50
    return edge_mask                                                    # dòng 51
```
`np.clip` **trước** khi ép kiểu (dòng 47) là bắt buộc: độ lớn gradient thường
xuyên vượt 255, ép kiểu trực tiếp sẽ **tràn quay vòng** (260 → 4), biến biên
mạnh nhất thành đen — đúng ngược lại ý định.

`_, edge_mask = cv2.threshold(...)`: hàm trả tuple `(ngưỡng_đã_dùng, ảnh)`. Ta
đã biết ngưỡng nên bỏ qua bằng `_`.

Lưu ý mặc định ở đây là **50**, nhưng `detect_vehicle_candidates` mặc định **40**
và `config.SOBEL_EDGE_THRESHOLD` cũng **40**. Con số 50 là **code chết** trong
luồng gọi hiện tại; giá trị thực sự chạy là 40 (khớp chú thích Hình 3.2).

#### c) Trích vùng ứng viên — dòng 54–81

```python
def detect_vehicle_candidates(gray_img, min_area=500, max_area=150000,  # dòng 54
                              max_aspect_ratio=4.0, edge_threshold=40): # dòng 55
    binary_mask, threshold = iterative_global_threshold(gray_img)       # dòng 57
    edge_mask     = sobel_edge_detection(gray_img, edge_threshold=edge_threshold)  # dòng 58
    combined_mask = cv2.bitwise_or(binary_mask, edge_mask)              # dòng 59
```
Dùng **OR** chứ không phải AND: cường độ và biên thất bại ở những tình huống
*khác nhau* (xe tối đều màu không có biên trong; mặt đường nhiều texture có biên
nhưng không tương phản cường độ). Phép hợp khôi phục được nhiều vật thể hơn phép
giao. Cái giá: nhiều false positive hơn — đúng như Hình 3.2 minh hoạ.

```python
    kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))    # dòng 61
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)      # dòng 62
```
Giãn nở làm mọi vùng trắng "phình" ra ~2 px mỗi hướng. **Vì sao cần:** Sobel tạo
ra *đường viền* — các vòng mảnh và đứt đoạn. Không giãn nở thì vòng viền quanh
một chiếc xe vỡ thành 5 contour riêng, và bộ lọc diện tích sẽ loại hết vì mỗi
mảnh quá nhỏ. Giãn nở hàn chúng thành một khối.

```python
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # dòng 63-65
```
- `RETR_LIST` — trả *toàn bộ* contour theo danh sách phẳng, kể cả contour lồng
  bên trong contour khác.
- `CHAIN_APPROX_SIMPLE` — chỉ lưu điểm góc của đoạn thẳng, không lưu từng pixel
  biên. Hình chữ nhật còn 4 điểm thay vì 400 → nhẹ bộ nhớ, nhanh hơn.

```python
    min_aspect_ratio = 1.0 / max_aspect_ratio                           # dòng 69 — 4.0 → 0.25
    bounding_boxes = []                                                 # dòng 71
    for contour in contours:                                            # dòng 72
        area = cv2.contourArea(contour)                                 # dòng 73
        if area <= min_area or area >= max_area:                        # dòng 74
            continue                                                    # dòng 75
        x, y, width, height = cv2.boundingRect(contour)                 # dòng 77
        aspect_ratio = width / float(max(height, 1))                    # dòng 78
        if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:        # dòng 79
            bounding_boxes.append([x, y, x + width, y + height])        # dòng 80
    return bounding_boxes, combined_mask, threshold                     # dòng 81
```
Dòng 69: chặn dưới được **suy ra** từ chặn trên, không phải hằng số riêng. Nên
sửa `MAX_ASPECT_RATIO` trong config là cả hai phía đều đổi theo.

Dòng 74: `continue` nhảy sang contour kế tiếp, bỏ qua phần còn lại của thân vòng lặp.

Dòng 77: `cv2.boundingRect` trả hộp trục song song nhỏ nhất bao contour, dạng
`(x, y, w, h)` — đúng công thức §2.4.3 (`x_min, y_min, w, h`).

Dòng 78: `max(height, 1)` chống chia cho 0 khi contour chỉ dày 1 dòng pixel.
`float(...)` để Python 3 chia thật (thực ra Python 3 đã chia thật, dòng này chỉ
là phòng thủ rõ ý).

Dòng 79 — **bộ lọc tỉ lệ hai phía** (§3.3): chặn trên 4.0 loại vạch kẻ đường và
dải phân cách (rất rộng, rất dẹt); chặn dưới 0.25 loại cột điện và thân cây (rất
cao, rất hẹp).

Dòng 80 — phép đổi `xywh → xyxy` là **có chủ đích**: làm nhánh truyền thống xuất
ra cùng định dạng với YOLO, nhờ vậy `visualization.py` không cần rẽ nhánh.

**Hợp đồng đầu ra:** danh sách box ứng viên. **Không có lớp, không có
confidence.** Đây là lý do Dashboard hiển thị "Vùng ứng viên" thay vì 4 ô đếm, và
lý do CSV ghi `class_name="candidate", confidence="N/A"`.

### 2.4 `optical_flow.py` — Lucas–Kanade (Listing 3.3)

```python
class LucasKanadeTracker:                                               # dòng 15
    def __init__(self):                                                 # dòng 16
        self.feature_params = dict(                                     # dòng 17
            maxCorners=100,      # dòng 18 — tối đa 100 điểm. Đây là flow THƯA, không dày đặc.
            qualityLevel=0.3,    # dòng 19 — chỉ giữ góc có điểm số ≥ 0.3 × điểm số góc tốt nhất
            minDistance=7,       # dòng 20 — ép khoảng cách ≥7 px, tránh dồn cục vào một vật
            blockSize=7,         # dòng 21 — cửa sổ tính điểm số Shi-Tomasi
        )
        self.lk_params = dict(                                          # dòng 23
            winSize=(15, 15),    # dòng 24 — vùng lân cận n pixel tạo hệ Av = b (§2.5.2)
            maxLevel=2,          # dòng 25 — 3 tầng pyramid (0,1,2)
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),  # dòng 26-30
        )
        self.prev_points = None                                         # dòng 32 — TRẠNG THÁI giữ qua các frame
```
Dòng 32 là lý do đây phải là **class** chứ không phải hàm rời: tập điểm phải sống
sót qua các lần gọi.

`winSize=(15,15)` (dòng 24): đây chính là "cửa sổ n pixel" ở §2.5.2 — 15×15 = 225
phương trình cho 2 ẩn `(u,v)`, giải bằng bình phương tối thiểu.

`maxLevel=2` (dòng 25) — **vì sao quan trọng:** Lucas-Kanade thuần giả định
chuyển động *nhỏ* (§2.5.1). Xe dịch 40 px giữa hai frame phá vỡ giả định đó.
Pyramid giải flow ở ảnh thu nhỏ ¼ trước (nơi 40 px trông như 10 px), rồi tinh
chỉnh dần xuống. Đây là phương pháp Bouguet, tài liệu [6].

`criteria` (dòng 26–30): dừng tinh chỉnh lặp tại mỗi điểm khi đạt 10 vòng **HOẶC**
sai số < 0.03 px — cái nào tới trước. `|` là phép OR bit, gộp hai cờ điều kiện.

```python
    def reset(self):                                                    # dòng 34
        self.prev_points = None                                         # dòng 36
```
Buộc gieo lại điểm ở frame kế tiếp.

```python
    def track_features(self, prev_gray, curr_gray):                     # dòng 38
        if self.prev_points is None or len(self.prev_points) == 0:      # dòng 40
            self.prev_points = cv2.goodFeaturesToTrack(                 # dòng 41
                prev_gray, mask=None, **self.feature_params)            # dòng 42
```
**Gieo lười:** Shi–Tomasi chỉ chạy khi tập điểm đã cạn. §2.5.2 giải thích lý do —
vùng phẳng cho AᵀA gần suy biến (không giải được), nên phải chọn điểm góc nơi
AᵀA có điều kiện tốt.

`**self.feature_params` là **dict unpacking**: bung dict thành các tham số có
tên, tương đương gõ `maxCorners=100, qualityLevel=0.3, ...`.

```python
        if self.prev_points is None or len(self.prev_points) == 0:      # dòng 45
            return np.empty((0, 2)), np.empty((0, 2))                   # dòng 46
```
Kiểm tra **lần hai** vì `goodFeaturesToTrack` **có thể trả `None`** khi ảnh không
có góc nào (ví dụ bức tường trắng). Trả về hai mảng rỗng shape `(0,2)` thay vì
`None`, để nơi gọi chỉ cần `len(points) > 0` mà không phải kiểm `is None`.

```python
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(              # dòng 48
            prev_gray, curr_gray, self.prev_points, None, **self.lk_params)  # dòng 49-53
        if curr_points is None or status is None:                       # dòng 55
            self.reset()                                                # dòng 56
            return np.empty((0, 2)), np.empty((0, 2))                   # dòng 57
```
Hàm trả 3 giá trị `(điểm_mới, status, error)`; ta bỏ `error` bằng `_`.

```python
        valid       = status.ravel() == 1                               # dòng 59
        valid_prev  = self.prev_points.reshape(-1, 2)[valid]            # dòng 60
        valid_curr  = curr_points.reshape(-1, 2)[valid]                 # dòng 61
        motion_vectors = valid_curr - valid_prev                        # dòng 62 — v = p(t+1) − p(t)
        self.prev_points = valid_curr.reshape(-1, 1, 2)                 # dòng 64
        return valid_curr, motion_vectors                               # dòng 65
```
Dòng 59: `status` có shape `(N,1)`; `.ravel()` làm phẳng thành `(N,)` để so sánh
tạo mask boolean đúng chiều. `status[i]==1` nghĩa là điểm i được tìm thấy ở frame
mới; `==0` nghĩa là mất (bị che, ra khỏi khung, hết texture).

Dòng 60–61: `.reshape(-1, 2)` đổi `(N,1,2)` → `(N,2)`. Số `-1` nghĩa là "tự tính
chiều này". Rồi `[valid]` lọc boolean.

Dòng 62: phép trừ mảng NumPy làm theo từng phần tử — ra đúng vector `(u,v)` của §2.5.2.

Dòng 64: đổi ngược `(N,2)` → `(N,1,2)` vì `calcOpticalFlowPyrLK` ở lần gọi sau
**bắt buộc** shape đó.

**Hiện tượng hao hụt điểm:** mỗi frame đều loại các điểm bám hỏng (dòng 60–61),
và không có gì gieo lại cho đến khi tập điểm về **đúng bằng 0** (dòng 40). Nên số
mũi tên giảm dần theo thời lượng video. Đây đúng là giới hạn báo cáo nêu ở §3.4.1:
module ước lượng chuyển động của *điểm*, **không** duy trì ID phương tiện. Các
điểm không hề được gán vào box ứng viên ở mục 2.3.

### 2.5 `yolo_detector.py` — nhánh học sâu (Listing 3.4)

```python
import torch                                                            # dòng 3
from ultralytics import YOLO                                            # dòng 4

class Yolo11VehicleDetector:                                            # dòng 7
    VEHICLE_NAMES = {"car","motorcycle","motorbike","bus","truck","container truck"}  # dòng 8-15
```
Dùng `set` (dấu `{}` không có `:`) → kiểm tra thành viên **O(1)**, chạy một lần
cho mỗi box mỗi frame. Nếu dùng `list` thì mỗi lần kiểm phải quét tuần tự.

Đây là **biến của class**, không phải của instance — mọi đối tượng dùng chung một
bản, không nhân bản theo từng instance.

Lọc theo **tên** thay vì COCO ID `{2,3,5,7}` chính là thứ giúp model tự huấn
luyện (thứ tự lớp khác COCO) chạy được không cần sửa (§3.5, §3.5.3).

```python
    def __init__(self, model_path="yolo11n.pt"):                        # dòng 17
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # dòng 18
        self.model  = YOLO(model_path)                                  # dòng 19
        self.model.to(self.device)                                      # dòng 20
```
Dòng 18: tự chọn GPU nếu có, không thì CPU. `"cuda:0"` là GPU thứ nhất.
Dòng 19: `YOLO(...)` tự tải trọng số từ mạng ở lần dùng đầu nếu file chưa có.
Dòng 20: chuyển tham số sang thiết bị **một lần**. Đây là bước tốn kém nhất →
phải khởi tạo ngoài vòng lặp frame.

```python
    def detect_frame(self, bgr_frame, conf_threshold=0.25,              # dòng 22
                     iou_threshold=0.45, max_det=300):                  # dòng 23
        if bgr_frame is None or bgr_frame.size == 0:                    # dòng 24
            raise ValueError("Khung hình đầu vào rỗng.")                # dòng 25
        results = self.model.predict(                                   # dòng 27
            bgr_frame,           # dòng 28 — BGR thô; Ultralytics tự lo BGR→RGB, letterbox, /255, HWC→CHW
            conf=conf_threshold, # dòng 29 — cổng lọc confidence, áp dụng TRƯỚC NMS
            iou=iou_threshold,   # dòng 30 — ngưỡng loại bỏ của NMS
            max_det=max_det,     # dòng 31 — trần số box còn lại
            device=self.device,  # dòng 32
            verbose=False,       # dòng 33 — tắt log in ra console mỗi frame
        )
```
**Toàn bộ Chương 2.6 xảy ra bên trong đúng lời gọi dòng 27 này:** letterbox về
640×640 → Backbone (Conv/C3k2/SPPF/C2PSA) → Neck (FPN+PAN) → Detect head → giải
mã DFL trên 8400 vị trí lưới → sigmoid confidence → lọc conf → NMS → quy đổi toạ
độ về frame gốc.

```python
        detected_vehicles = []                                          # dòng 35
        for result in results:                                          # dòng 37 — mỗi ảnh vào cho 1 Results;
            if result.boxes is None:                                    # dòng 38   ở đây luôn dài đúng 1
                continue                                                # dòng 39 — frame không phát hiện được gì
            for box in result.boxes.cpu():                              # dòng 41
```
`.cpu()` (dòng 41) chuyển tensor từ GPU về RAM host **một lần cho cả frame**. Nếu
gọi `.cpu()` bên trong vòng lặp box thì mỗi box một lần chuyển → rất chậm.

```python
                class_id   = int(box.cls[0])                            # dòng 42
                class_name = str(self.model.names[class_id])            # dòng 43
                if class_name.strip().lower() not in self.VEHICLE_NAMES:  # dòng 44
                    continue                                            # dòng 45
```
Dòng 42: `box.cls` là tensor shape `(1,)`; `[0]` lấy phần tử, `int()` đổi sang số
Python thường.
Dòng 43: `model.names` là dict `{class_id: tên}` do **chính model** mang theo —
nên model custom tự khai báo tên của nó.
Dòng 44: `.strip()` bỏ khoảng trắng thừa, `.lower()` đưa về chữ thường, để `"Car"`,
`"car "` và `"CAR"` đều khớp.

```python
                detected_vehicles.append({                              # dòng 47
                    "bbox":       box.xyxy[0].numpy().astype(int).tolist(),  # dòng 48
                    "class":      class_name,                           # dòng 49
                    "confidence": float(box.conf[0]),                   # dòng 50
                })
        return detected_vehicles                                        # dòng 53
```
Dòng 48 là **ranh giới** nơi kiểu Ultralytics dừng lại và Python thuần bắt đầu:
`tensor → ndarray → int → list`. Mọi thứ phía sau (vẽ, đếm, xuất CSV/JSON) chỉ
tiêu thụ dict này, không biết gì về Ultralytics.

### 2.6 `visualization.py` — vẽ kết quả

```python
def draw_classical_boxes(frame, boxes):                                 # dòng 15
    output = frame.copy()                                               # dòng 17
    for x1, y1, x2, y2 in boxes:                                        # dòng 18
        cv2.rectangle(output, (x1,y1), (x2,y2), config.BOX_COLOR, config.BOX_THICKNESS)  # dòng 19-21
    return output                                                       # dòng 22
```
`frame.copy()` (dòng 17) là bắt buộc: mảng NumPy truyền theo **tham chiếu**. Vẽ
thẳng lên `frame` sẽ sửa luôn biến của hàm gọi — và ở `pipeline.py` dòng 86,
`frame` còn được dùng để tính `curr_gray`, nên sẽ tính trên ảnh đã có box vẽ đè.

Dòng 18: giải nén trực tiếp 4 giá trị từ mỗi box `[x1,y1,x2,y2]`.

```python
def draw_yolo_detections(frame, detections):                            # dòng 25
    output = frame.copy()                                               # dòng 27
    for det in detections:                                              # dòng 28
        x1, y1, x2, y2 = det["bbox"]                                    # dòng 29
        label = f"{det['class']} {det['confidence']:.2f}"               # dòng 30 — ví dụ "car 0.87"
        cv2.rectangle(output, (x1,y1), (x2,y2), config.BOX_COLOR, config.BOX_THICKNESS)  # dòng 31-33
        cv2.putText(output, label, (x1, max(y1 - 8, 0)),                # dòng 34-36
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.BOX_COLOR, 2) # dòng 37-41
    return output                                                       # dòng 43
```
Dòng 30: `:.2f` làm tròn 2 chữ số thập phân.
Dòng 36: `max(y1 - 8, 0)` — đặt chữ **trên** box 8 px, nhưng chặn ở 0 để box sát
mép trên không bị vẽ chữ ra ngoài ảnh (toạ độ âm → OpenCV không vẽ gì).

Hàm này chỉ chạm `det["bbox"]/["class"]/["confidence"]` — toàn kiểu Python thuần.
Đó chính là ý của §3.5.1: khối hiển thị không bao giờ thấy tensor Ultralytics.

```python
def draw_motion_vectors(frame, points, motion_vectors, display_scale=5.0):  # dòng 46
    output = frame.copy()                                               # dòng 58
    for (xc, yc), (u, v) in zip(points, motion_vectors):                # dòng 59
        p_curr = (int(xc), int(yc))                                     # dòng 60
        p_prev = (int(xc - u * display_scale), int(yc - v * display_scale))  # dòng 61
        cv2.arrowedLine(output, p_prev, p_curr, (0,0,255), 2, tipLength=0.4)  # dòng 62
        cv2.circle(output, p_curr, 3, (255,0,0), -1)                    # dòng 63
    return output                                                       # dòng 64
```
`zip()` (dòng 59) ghép hai mảng, lặp đồng thời từng cặp `(điểm, vector)`.

**Vì sao có `display_scale=5.0` (thêm sau khi phát hiện mũi tên vô hình
trên video thật):** thử nghiệm tổng hợp với chuyển động 4 px/frame — con số
thực tế cho video 25-30 FPS — cho thấy `motion_vectors` được Lucas-Kanade
tính **đúng** (đo lại được đúng 4.00 px), nhưng khi vẽ đúng tỉ lệ 1:1, mũi
tên dài 4 px bị chấm tròn bán kính 3 px ở dòng 63 che gần hết, và Dashboard
còn thu nhỏ khung hình về 480p trước khi hiển thị (§3.7) khiến nó nhỏ hơn
nữa — kết quả là người dùng chỉ thấy chấm xanh, không thấy mũi tên đỏ nào,
dù thuật toán chạy hoàn toàn đúng. Đây thuần túy là vấn đề **hiển thị**, không
phải lỗi tính toán.

Dòng 61 nhân `u`, `v` với `display_scale` **chỉ** khi tính điểm đuôi mũi
tên `p_prev` — phóng đại độ dài mũi tên lên gấp 5 lần cho dễ nhìn. Điểm đầu
mũi tên `p_curr` (dòng 60, cũng là tâm chấm tròn) **không đổi**, vẫn đúng vị
trí thật 100%. Giá trị `motion_vectors` mà hàm này *nhận vào* không hề bị
sửa — việc phóng đại chỉ xảy ra cục bộ trong phép vẽ, không lan ra bất kỳ
biến nào khác. Vì CSV/JSON xuất ra (§3.7) không chứa dữ liệu chuyển động
(chỉ có `frame_id, class_name, confidence, count`), việc phóng đại này
**không ảnh hưởng đến bất kỳ số liệu báo cáo nào**, chỉ ảnh hưởng phần vẽ.

Dòng 61 — logic hay khác: hàm nhận vị trí **hiện tại** rồi *dựng ngược* vị trí trước
bằng cách trừ vector chuyển động. Nhờ vậy mũi tên vẽ từ `p_prev` → `p_curr`, tức
chỉ đúng **hướng di chuyển**.

Dòng 62: `(0,0,255)` là **đỏ** trong BGR. `tipLength=0.4` = đầu mũi tên dài 40%
thân.
Dòng 63: `(255,0,0)` là **xanh dương** trong BGR. Tham số `-1` cuối = tô đặc hình
tròn (số dương sẽ là độ dày viền).

### 2.7 `__init__.py` — khai báo package

```python
from . import preprocessing         # dòng 9  — import sẵn để `from transdetect import preprocessing`
from . import classical_detector    # dòng 10   chạy được mà không cần đường dẫn đầy đủ
...
__all__ = ["preprocessing", "classical_detector", ...]   # dòng 16-23
__version__ = "0.1.0"                                    # dòng 25
```
`__all__` khai báo những tên được xuất khi ai đó viết `from transdetect import *`.
File này chạy **tự động** khi package được import lần đầu.

---

## Phần 3 — Kịch bản B: Dashboard (`app_streamlit.py`)

### Điều phải hiểu trước tiên về Streamlit

Streamlit **chạy lại toàn bộ file từ dòng 1 đến dòng cuối** mỗi khi người dùng
tương tác (bấm nút, kéo slider). Không có "sự kiện onClick" như web thường. Hệ quả:

- Mọi biến thường bị **xoá sạch** sau mỗi lần chạy lại.
- Thứ cần sống sót phải nằm trong `st.session_state` (một dict tồn tại xuyên các lần chạy).
- Vòng lặp frame nằm ở **cuối** file, và trong lúc nó chạy thì UI phía trên đã
  được dựng xong; các `st.empty()` placeholder được cập nhật từ bên trong vòng lặp.

### 3.1 Import và cấu hình — dòng 1–30

```python
import streamlit as st                                              # dòng 7
import pandas as pd                                                 # dòng 8  — chỉ dùng để xuất CSV
import cv2, time, tempfile, os, sys, json                           # dòng 9-14
sys.path.append(os.path.dirname(os.path.abspath(__file__)))         # dòng 17
from src.transdetect import config, preprocessing, classical_detector, visualization  # dòng 18
from src.transdetect.optical_flow import LucasKanadeTracker         # dòng 19
from src.transdetect.yolo_detector import Yolo11VehicleDetector     # dòng 20
```
Dòng 17: thêm thư mục chứa chính file này vào đường dẫn tìm module, để `import
src.transdetect` chạy được kể cả khi gọi từ thư mục khác. `__file__` là đường dẫn
file hiện tại, `abspath` đổi thành tuyệt đối, `dirname` lấy thư mục cha.

```python
st.set_page_config(page_title="TransDetect-Vid", page_icon="🚗",     # dòng 25-30
                   layout="wide", initial_sidebar_state="collapsed")
```
**Bắt buộc là lệnh Streamlit đầu tiên** trong file, gọi sau bất kỳ lệnh `st.*`
nào khác sẽ lỗi. `layout="wide"` cho phép dùng hết chiều ngang màn hình — cần
thiết cho bố cục 3 cột.

### 3.2 Khởi tạo trạng thái — dòng 32–48

```python
if "is_running" not in st.session_state:                            # dòng 35
    st.session_state.is_running = False                             # dòng 36
if "is_paused" not in st.session_state: ...                         # dòng 37-38
if "current_frame_idx" not in st.session_state: ...                 # dòng 39-40
if "uploaded_video_path" not in st.session_state: ...               # dòng 41-42
if "uploaded_file_name" not in st.session_state: ...                # dòng 43-44
if "results_history" not in st.session_state: ...                   # dòng 45-46
if "export_data_list" not in st.session_state: ...                  # dòng 47-48
```
Mẫu `if "x" not in ...` (chứ không gán thẳng) là **bắt buộc**: gán thẳng sẽ reset
giá trị về mặc định ở **mỗi lần** chạy lại, tức mỗi lần bấm nút — và bạn mất
sạch tiến trình.

Bảy biến trạng thái:

| Biến | Vai trò |
|---|---|
| `is_running` | Vòng lặp frame có đang chạy không |
| `is_paused` | Đang tạm dừng |
| `current_frame_idx` | Frame đang ở đâu, để Resume tua lại đúng chỗ |
| `uploaded_video_path` | Đường dẫn file video thật trên đĩa |
| `uploaded_file_name` | Tên hiển thị, đồng thời dùng để phát hiện "đổi video" |
| `results_history` | 6 dòng gần nhất của bảng kết quả |
| `export_data_list` | Toàn bộ dữ liệu để xuất CSV/JSON |

### 3.3 CSS — dòng 50–125

`CUSTOM_CSS` là một chuỗi HTML `<style>`. Vài luật đáng chú ý:

```css
#MainMenu, footer, header { visibility: hidden; }        /* dòng 57 — ẩn thanh menu mặc định của Streamlit */
[data-testid="collapsedControl"] { display: none; }      /* dòng 58 — ẩn nút mở sidebar */
```
Phần còn lại định nghĩa các class dùng ở mục 3.4: `.header-bar`, `.vehicle-card`,
`.fps-value`, `.results-table`, `.video-controls`.

### 3.4 Các hàm dựng HTML — dòng 128–195

#### `_build_results_table()` — dòng 131–152

```python
def _build_results_table(rows, highlight_row_idx=0):                # dòng 131
    if not rows:                                                    # dòng 132
        return '<div ...>No data yet. Run detection.</div>'         # dòng 133
```
Dòng 132: `not rows` đúng khi list rỗng — trả về ô trống thay vì bảng không có dòng nào.

```python
    html = '<table class="results-table"><thead><tr>'               # dòng 135
    html += '<th rowspan="2" ...>Frame</th>'                        # dòng 136
    html += '<th rowspan="2" ...>Time (s)</th>'                     # dòng 137
    for name in ("Car","Motorcycle","Bus","Truck"):                 # dòng 138
        html += f'<th colspan="2" class="group-header">{name}</th>' # dòng 139
    html += '<th rowspan="2" ...>Total</th></tr><tr>'               # dòng 140
    for _ in range(4):                                              # dòng 141
        html += '<th class="sub-header">Count</th><th class="sub-header">Avg Conf</th>'  # dòng 142
    html += "</tr></thead><tbody>"                                  # dòng 143
```
Đây là bảng **header hai tầng**: `rowspan="2"` cho cột đơn (Frame, Time, Total)
kéo dài qua cả hai hàng tiêu đề; `colspan="2"` cho mỗi loại xe chiếm 2 cột con
(Count + Avg Conf).

```python
    for i, r in enumerate(rows):                                    # dòng 145
        cls = ' class="row-highlight"' if i == highlight_row_idx else ""  # dòng 146
        html += f"<tr{cls}><td>{r[0]}</td><td>{r[1]}</td>"          # dòng 147
        for j in range(2, 10, 2):                                   # dòng 148
            html += f"<td>{r[j]}</td><td>{r[j+1]:.2f}</td>"         # dòng 149
        html += f"<td>{r[10]}</td></tr>"                            # dòng 150
```
Dòng 148: `range(2, 10, 2)` sinh `2, 4, 6, 8` — chỉ số của 4 cặp
`(count, avg_conf)` trong tuple 11 phần tử. `r[0]`=frame, `r[1]`=time,
`r[2..9]`=4 cặp, `r[10]`=total.

Dòng 146: dòng mới nhất (chỉ số 0) được tô sáng.

#### `_avg_conf()` — dòng 154–158

```python
def _avg_conf(detections, display_name):                            # dòng 154
    values = [d["confidence"] for d in detections                   # dòng 156
              if d.get("display_class") == display_name]            # dòng 157
    return sum(values) / len(values) if values else 0.0             # dòng 158
```
List comprehension lọc theo `display_class` — trường được gắn thêm ở dòng 488
(xem mục 3.7). Dòng 158 chống chia cho 0 khi lớp đó không có box nào.

Với nhánh Classical, `detections` luôn rỗng nên hàm này luôn trả `0.0` — đó là lý
do bảng kết quả của nhánh truyền thống hiển thị toàn số 0.

#### `render_vehicle_card()` / `render_video_info()` / `render_header()` — dòng 160–195

Ba hàm chỉ sinh chuỗi HTML từ f-string, không có logic. `render_video_info` dòng
171 tính phần trăm hoàn thành, có chống chia 0.

### 3.5 Cột 1 — Control Panel (dòng 206–283)

```python
col1, col2, col3 = st.columns([1, 2, 1], gap="medium")              # dòng 204
```
Tỉ lệ chiều rộng 1:2:1 — cột giữa gấp đôi hai bên, đúng `UI_REQUIREMENTS.md`.

```python
    test_videos_dir = os.path.join(os.path.dirname(__file__), "test_videos")  # dòng 211
    if not os.path.exists(test_videos_dir):                         # dòng 212
        try: os.makedirs(test_videos_dir, exist_ok=True)            # dòng 213
        except: pass                                                # dòng 214
    local_videos = ["-- Select a local video --"]                   # dòng 216
    if os.path.exists(test_videos_dir):                             # dòng 217
        local_videos += [f for f in os.listdir(test_videos_dir)      # dòng 218
                         if f.lower().endswith(('.mp4','.avi','.mov'))]
    selected_local = st.selectbox("Select from 'test_videos' folder", local_videos)  # dòng 220
```
Dòng 216: phần tử đầu là một mục giả làm giá trị "chưa chọn" — Streamlit
`selectbox` luôn chọn phần tử đầu tiên mặc định.
Dòng 218: `.lower()` để `.MP4` viết hoa cũng nhận.

```python
    uploaded_file = st.file_uploader("Upload new video", type=["mp4","avi","mov"])  # dòng 225
    if uploaded_file is not None:                                   # dòng 228
        if st.session_state.uploaded_file_name != uploaded_file.name:  # dòng 229
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")  # dòng 230
            tfile.write(uploaded_file.read())                       # dòng 231
            st.session_state.uploaded_video_path = tfile.name       # dòng 232
            st.session_state.uploaded_file_name  = uploaded_file.name  # dòng 233
            st.session_state.results_history  = []                  # dòng 234
            st.session_state.export_data_list = []                  # dòng 235
            st.session_state.is_running = False                     # dòng 236
```
Dòng 229 là **cực kỳ quan trọng**: vì script chạy lại liên tục, `uploaded_file`
vẫn còn đó ở mọi lần chạy. Không có phép so tên này thì mỗi lần bấm nút sẽ ghi
lại file tạm và xoá sạch kết quả.

Dòng 230: `delete=False` giữ file sau khi đóng — OpenCV cần đọc nó về sau.
Dòng 234–236: đổi video thì xoá kết quả cũ, tránh trộn số liệu hai video.

```python
    st.radio("method", ["Classical Pipeline", "YOLO11"], ..., key="method_selector")  # dòng 247
    st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01, key="conf_thresh")        # dòng 251
    st.slider("IoU Threshold (NMS)",  0.0, 1.0, 0.45, 0.01, key="iou_thresh")         # dòng 252
    st.number_input("Max Detection per Frame", 1, 500, config.MAX_DET, 10, key="max_det")  # dòng 253
    for _name in config.DISPLAY_CLASSES:                                              # dòng 259
        st.checkbox(f"{_icons[_name]}  {_name}", value=True, key=f"cls_{_name}")      # dòng 260
```
**`key=` chính là dây nối.** `key="conf_thresh"` khiến Streamlit tự ghi giá trị
slider vào `st.session_state.conf_thresh`; vòng lặp frame đọc lại ở dòng 473. Đây
là cơ chế duy nhất để widget ở cột 1 nói chuyện được với vòng lặp ở cuối file.

Dòng 259–260 sinh 4 checkbox với key `cls_Car`, `cls_Motorcycle`, `cls_Bus`,
`cls_Truck` — trùng đúng tên trong `CLASS_ALIAS`, nên dòng 486 tra ngược được.

```python
    if st.button("▶ Run", type="primary", use_container_width=True):  # dòng 265
        if st.session_state.uploaded_video_path:                      # dòng 266
            st.session_state.is_running = True                        # dòng 267
            st.session_state.is_paused  = False                       # dòng 268
            st.session_state.current_frame_idx = 0                    # dòng 269
            st.session_state.results_history   = []                   # dòng 270
            st.session_state.export_data_list  = []                   # dòng 271
        else:
            st.error("Upload a video first!")                         # dòng 273
```
`st.button` trả `True` **chỉ ở đúng lần chạy lại ngay sau khi bấm**, các lần sau
trả `False`. Nên phải ghi cờ vào `session_state` (dòng 267) chứ không thể dựa vào
giá trị nút.

Nút Pause (dòng 276–278) đảo cờ `is_paused`; nút Stop (dòng 280–283) tắt cờ chạy
và đưa `current_frame_idx` về 0.

### 3.6 Cột 2 & 3 — dựng placeholder (dòng 285–364)

```python
    time_text_placeholder   = st.empty()                            # dòng 291
    progress_bar_placeholder= st.empty()                            # dòng 292
    video_placeholder       = st.empty()                            # dòng 293
    controls_placeholder    = st.empty()                            # dòng 294
```
`st.empty()` tạo một **ô trống có thể ghi đè**. Vòng lặp frame sẽ liên tục gọi
`video_placeholder.image(...)` để thay ảnh **tại chỗ** thay vì đẩy thêm ảnh mới
xuống dưới. Không có kỹ thuật này thì mỗi frame sẽ nối thêm một ảnh và trang dài
vô hạn.

```python
    if st.session_state.get("export_data_list"):                    # dòng 308
        df = pd.DataFrame(st.session_state.export_data_list)        # dòng 309
        csv_data = df.to_csv(index=False)                           # dòng 310
    else:
        csv_data = "frame_id,class_name,confidence,count\n"         # dòng 312
    ...
    st.download_button("⬇ CSV", csv_data, "res.csv", ...)           # dòng 315
    json_data = json.dumps(st.session_state.get("export_data_list", []),  # dòng 316
                           ensure_ascii=False, indent=2)
    st.download_button("{ } JSON", json_data, "res.json", ...)      # dòng 317
```
Dòng 312: khi chưa có dữ liệu vẫn phải trả file CSV có header, nếu không nút tải
sẽ cho file rỗng hoàn toàn.
Dòng 309: `pd.DataFrame` nhận list các dict → tự lấy khoá làm tên cột.
Dòng 316: `ensure_ascii=False` giữ nguyên tiếng Việt có dấu; `indent=2` cho JSON
dễ đọc.

Cột 3 (dòng 323–364) dựng thêm các placeholder cho FPS, 4 ô đếm xe, thông tin
video và thanh tiến trình. Khối `if not st.session_state.is_running` (dòng 347)
đổ giá trị mặc định 0 vào chúng khi chưa chạy.

Đáng chú ý dòng 352–356: khi chọn Classical, chỉ hiện **một** ô "Vùng ứng viên",
ba ô còn lại gọi `.empty()` để **xoá** — vì nhánh này không phân loại được xe.

### 3.7 Vòng lặp frame — dòng 367–589 (phần lõi)

```python
    if st.session_state.is_running and st.session_state.uploaded_video_path:  # dòng 370
        cap = cv2.VideoCapture(st.session_state.uploaded_video_path)          # dòng 371
        fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30.0                      # dòng 372
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))                 # dòng 373
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                            # dòng 374
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                           # dòng 375
        duration_s = total_frames / fps_video if fps_video > 0 else 0         # dòng 377
        dur_str = time.strftime('%H:%M:%S', time.gmtime(duration_s))          # dòng 378
```
Dòng 378: `time.gmtime` đổi số giây thành struct thời gian UTC, `strftime` định
dạng thành `HH:MM:SS`. Dùng `gmtime` chứ không `localtime` để không bị lệch múi giờ.

```python
        if st.session_state.get("is_paused", False):                # dòng 382
            if st.session_state.current_frame_idx > 0:              # dòng 383
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_idx)  # dòng 384
            ret, frame = cap.read()                                 # dòng 385
            if ret: video_placeholder.image(frame, channels="BGR")  # dòng 386-387
            cap.release()                                           # dòng 388
            return                                                  # dòng 389
```
Khi Pause: tua tới frame đang dừng, hiện **một** frame đó rồi `return` — thoát
hẳn `main()`. Không `return` thì vòng lặp vẫn chạy và trình duyệt bị chiếm.

```python
        if st.session_state.get("current_frame_idx", 0) > 0:        # dòng 391
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_idx)  # dòng 392
        frame_idx          = st.session_state.get("current_frame_idx", 0)  # dòng 394
        start_frame_idx    = frame_idx                              # dòng 395 — mốc để tính FPS trung bình
        start_time         = time.time()                            # dòng 396
        last_ui_update_time= 0                                      # dòng 397
        last_fps_time      = time.time()                            # dòng 398
        last_fps_frame_idx = frame_idx                              # dòng 399
        instant_fps        = 0.0                                    # dòng 400
```
Dòng 391–392 cho phép **Resume**: tua thẳng tới frame đang dở.
Dòng 395: nhớ mốc bắt đầu, vì sau Resume `frame_idx` không bắt đầu từ 0 — FPS
trung bình phải tính trên số frame *của lần chạy này*.

```python
        lk_tracker = None                                           # dòng 405
        prev_gray  = None                                           # dòng 406
        if st.session_state.method_selector == "Classical Pipeline":  # dòng 407
            lk_tracker = LucasKanadeTracker()                       # dòng 408

        yolo_detector = None                                        # dòng 410
        if st.session_state.method_selector == "YOLO11":            # dòng 411
            try:
                yolo_detector = Yolo11VehicleDetector(config.DEFAULT_YOLO_MODEL)  # dòng 414
            except Exception as e:                                  # dòng 415
                st.error(f"Failed to load YOLO model: {e}")         # dòng 416
                st.session_state.is_running = False                 # dòng 417
```
Cả hai đối tượng khởi tạo **ngoài** vòng lặp vì cùng lý do: giữ trạng thái (LK)
và tránh nạp lại model mỗi frame (YOLO).

Dòng 415–417: nạp model có thể hỏng (không mạng, thiếu file) — bắt lỗi và dừng
sạch thay vì để traceback đổ ra giao diện.

```python
        while cap.isOpened() and st.session_state.is_running \      # dòng 419
              and not st.session_state.get("is_paused", False):
            ret, frame = cap.read()                                 # dòng 420
            if not ret:                                             # dòng 421
                st.session_state.is_running = False                 # dòng 422
                st.session_state.current_frame_idx = 0              # dòng 423
                break                                               # dòng 424
            frame_idx += 1                                          # dòng 426
            st.session_state.current_frame_idx = frame_idx          # dòng 427
```
Dòng 419 kiểm **ba** điều kiện mỗi vòng: video còn mở, cờ chạy còn bật, không bị
pause. Dòng 421–424: hết video → tự tắt và đưa chỉ số về 0 để lần Run sau chạy lại từ đầu.

```python
            if frame_idx - last_fps_frame_idx >= 10:                # dòng 432
                instant_fps = (frame_idx - last_fps_frame_idx) / (curr_time - last_fps_time)  # dòng 433
                last_fps_time      = curr_time                      # dòng 434
                last_fps_frame_idx = frame_idx                      # dòng 435
            current_fps = instant_fps if instant_fps > 0 else 0     # dòng 437
            avg_fps = (frame_idx - start_frame_idx) / elapsed if elapsed > 0 else 0.0  # dòng 440
```
**Hai chỉ số FPS khác nhau** (§3.7, §4.2.1):
- Dòng 432–433: **FPS (Current)** — cửa sổ trượt 10 frame gần nhất. Cửa sổ này
  tồn tại để lần khởi động model ở frame đầu không kéo con số xuống vĩnh viễn.
- Dòng 440: **Average FPS** — tổng frame chia tổng thời gian kể từ lúc bấm Run.

```python
            detections = []                                         # dòng 443
            if st.session_state.method_selector == "Classical Pipeline":  # dòng 444
                pre = preprocessing.preprocess_frame(frame, config.MEDIAN_KERNEL_SIZE)  # dòng 445
                boxes, _, _ = classical_detector.detect_vehicle_candidates(  # dòng 446
                    pre, min_area=config.MIN_CONTOUR_AREA,          # dòng 448
                    max_area=config.MAX_CONTOUR_AREA,               # dòng 449
                    max_aspect_ratio=config.MAX_ASPECT_RATIO,       # dòng 450
                    edge_threshold=config.SOBEL_EDGE_THRESHOLD)     # dòng 451
                candidate_count = len(boxes)                        # dòng 453
                counts = {n: 0 for n in config.DISPLAY_CLASSES}     # dòng 454 — cả 4 ô đều 0
                frame_to_draw = visualization.draw_classical_boxes(frame.copy(), boxes)  # dòng 455
```
Bốn tham số dòng 448–451 lấy hết từ `config.py`, **giống hệt** `pipeline.py` —
Dashboard và CLI không còn lệch tham số nhau.

Dòng 454: nhánh truyền thống không phân loại được xe, nên 4 ô đếm luôn bằng 0.
Số thật nằm ở `candidate_count`.

```python
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # dòng 460 — ảnh xám THÔ
                if prev_gray is not None and lk_tracker is not None:  # dòng 461
                    points, motion = lk_tracker.track_features(prev_gray, curr_gray)  # dòng 462
                    if len(points) > 0:                              # dòng 463
                        frame_to_draw = visualization.draw_motion_vectors(  # dòng 464
                            frame_to_draw, points, motion)
                prev_gray = curr_gray                                # dòng 467
```
Giống hệt logic ở `pipeline.py` dòng 86–91. Ba điểm nhắc lại: dùng ảnh xám **thô**
(không dùng `pre`, vì `equalizeHist` phá vỡ giả định độ sáng không đổi §2.5.1);
vẽ **đè** lên `frame_to_draw` đã có box; frame đầu không có mũi tên.

Sau khi Pause rồi Resume, script chạy lại nên `lk_tracker` và `prev_gray` được
tạo mới → frame đầu sau Resume cũng không có mũi tên, rồi tracker tự gieo lại điểm.

```python
            else:                                                   # dòng 468 — nhánh YOLO11
                counts = {n: 0 for n in config.DISPLAY_CLASSES}     # dòng 469
                if yolo_detector:                                   # dòng 470
                    detections = yolo_detector.detect_frame(        # dòng 471
                        frame,
                        st.session_state.conf_thresh,               # dòng 473 ← slider cột 1
                        st.session_state.iou_thresh,                # dòng 474 ← slider cột 1
                        st.session_state.max_det)                   # dòng 475 ← number_input cột 1
                    kept = []                                       # dòng 479
                    for det in detections:                          # dòng 480
                        cls_name = config.CLASS_ALIAS.get(          # dòng 481
                            str(det.get("class","")).strip().lower())  # dòng 482
                        if cls_name is None:                        # dòng 484
                            continue                                # dòng 485 — nhãn lạ → loại, không xếp nhầm ô
                        if not st.session_state.get(f"cls_{cls_name}", True):  # dòng 486
                            continue                                # dòng 487 — bị bỏ tick ở Target Classes
                        det["display_class"] = cls_name             # dòng 488 — gắn để _avg_conf() gom nhóm
                        counts[cls_name] += 1                       # dòng 489
                        kept.append(det)                            # dòng 490
                    detections = kept                               # dòng 491
                    frame_to_draw = visualization.draw_yolo_detections(frame.copy(), detections)  # dòng 493
```
Đây là nơi ba tham số ở cột 1 thực sự đi vào mô hình (dòng 473–475), và bốn
checkbox thực sự lọc kết quả (dòng 486). Đúng như §3.7 khẳng định.

Dòng 481–482: `.get()` trả `None` nếu không tìm thấy khoá — an toàn hơn `[...]`
sẽ ném `KeyError`.
Dòng 488: gắn thêm trường `display_class` để hàm `_avg_conf` (dòng 154) gom nhóm
theo tên hiển thị chứ không theo tên thô.

```python
            c_car   = counts.get("Car", 0)                          # dòng 497
            c_moto  = counts.get("Motorcycle", 0)                   # dòng 498
            c_bus   = counts.get("Bus", 0)                          # dòng 499
            c_truck = counts.get("Truck", 0)                        # dòng 500
            c_tot   = c_car + c_moto + c_bus + c_truck              # dòng 501
```

```python
            if st.session_state.method_selector == "Classical Pipeline":  # dòng 504
                st.session_state.export_data_list.append({          # dòng 506
                    "frame_id": frame_idx, "class_name": "candidate",  # dòng 507-508
                    "confidence": "N/A", "count": candidate_count})  # dòng 509-510
            else:                                                   # dòng 512
                for cls_name, count in counts.items():              # dòng 513
                    if count > 0:                                   # dòng 514 — chỉ ghi lớp có xuất hiện
                        avg_conf = round(_avg_conf(detections, cls_name), 2)  # dòng 515
                        st.session_state.export_data_list.append({...})       # dòng 516-521
```
Dòng 508–509: nhánh truyền thống ghi `class_name="candidate"` và
`confidence="N/A"` — trung thực, vì nó không phân loại được. Đây đúng như §3.7 mô tả.

Dòng 514: bỏ qua lớp có count 0 để file CSV không phình lên vô ích.

```python
            if curr_time - last_ui_update_time > 0.03:              # dòng 524 — tối đa ~33 lần/giây
```
**Vì sao phải giới hạn:** đẩy ảnh qua WebSocket là thao tác đắt. Cập nhật mỗi
frame trên video 60 FPS sẽ làm trình duyệt đơ. Toàn bộ khối 525–575 chỉ chạy khi
đã qua 30 ms kể từ lần vẽ trước.

```python
                h_ui, w_ui = frame_to_draw.shape[:2]                # dòng 533
                if h_ui > 480:                                      # dòng 534
                    scale = 480 / h_ui                              # dòng 535
                    frame_to_draw_ui = cv2.resize(frame_to_draw, (int(w_ui*scale), 480))  # dòng 536
                else:
                    frame_to_draw_ui = frame_to_draw                # dòng 538
                video_placeholder.image(frame_to_draw_ui, channels="BGR")  # dòng 540
```
Dòng 533: `.shape` của ảnh màu là `(H, W, 3)`; `[:2]` lấy hai phần tử đầu.
Dòng 535–536: thu nhỏ về cao 480 px, giữ nguyên tỉ lệ. Một PNG 1080p mỗi frame
sẽ làm nghẽn WebSocket.
Dòng 540: `channels="BGR"` báo cho Streamlit biết mảng theo thứ tự OpenCV — thiếu
tham số này thì màu bị đảo (xe đỏ thành xanh dương).

**Quan trọng:** chỉ ảnh *hiển thị* bị thu nhỏ. Việc phát hiện vẫn chạy trên
`frame` độ phân giải gốc.

```python
                if st.session_state.method_selector == "Classical Pipeline":  # dòng 549
                    car_placeholder.markdown(render_vehicle_card("Vùng ứng viên", candidate_count, "blue"), ...)  # dòng 550
                    moto_placeholder.empty(); bus_placeholder.empty(); truck_placeholder.empty()  # dòng 551-553
                else:
                    car_placeholder.markdown(render_vehicle_card("Car", c_car, "blue"), ...)      # dòng 555
                    ... (Motorcycle, Bus, Truck)                                                  # dòng 556-558
```
Dòng 550: nhãn ghi rõ "Vùng ứng viên" chứ không phải "Car" — đúng cảnh báo ở §3.7
và chú thích Hình 4.1: **không được đọc con số này là số ô tô**.

```python
                hist = (frame_idx, f"{curr_s:.2f}",                 # dòng 561
                        c_car,   _avg_conf(detections, "Car"),      # dòng 562
                        c_moto,  _avg_conf(detections, "Motorcycle"),  # dòng 563
                        c_bus,   _avg_conf(detections, "Bus"),      # dòng 564
                        c_truck, _avg_conf(detections, "Truck"), c_tot)  # dòng 565
                st.session_state.results_history.insert(0, hist)    # dòng 566
                if len(st.session_state.results_history) > 6:       # dòng 567
                    st.session_state.results_history.pop()          # dòng 568
```
Dòng 566: `insert(0, ...)` chèn vào **đầu** list → dòng mới nhất nằm trên cùng.
Dòng 568: `pop()` không tham số xoá phần tử **cuối** → giữ đúng 6 dòng gần nhất.
Kết hợp hai dòng này tạo ra một hàng đợi trượt.

```python
            if fps_video > 0:                                       # dòng 579
                expected_elapsed = (frame_idx - start_frame_idx) / fps_video  # dòng 580
                actual_elapsed   = time.time() - start_time         # dòng 581
                if actual_elapsed < expected_elapsed:               # dòng 582
                    time.sleep(expected_elapsed - actual_elapsed)   # dòng 583
```
**Điều tiết phát 1.0× thời gian thực.** Nếu xử lý nhanh hơn tốc độ gốc thì ngủ bù.

**Hệ quả cho Chương 4:** FPS hiển thị bị **chặn trần bởi FPS video nguồn** (~30).
Nó đo cả vòng lặp demo (đọc video + suy luận + vẽ + cập nhật UI), không phải
thông lượng suy luận thuần. Báo cáo đã thừa nhận ở ghi chú §4.2.1 ("giới hạn theo
FPS nguồn") — luôn đính kèm cảnh báo đó khi trích số.

```python
        cap.release()                                               # dòng 585
        if not st.session_state.is_running:                         # dòng 588
            st.rerun()                                              # dòng 589
```
Dòng 589: chạy lại script một lần cuối để nút tải CSV/JSON cập nhật với dữ liệu
đầy đủ (chúng được dựng ở dòng 308–317, tức **trước** vòng lặp).

```python
if __name__ == "__main__":                                          # dòng 592
    main()                                                          # dòng 593
```

---

## Phần 3.8 — Sửa lỗi tương phản chữ (theme + màu CSS)

Sau khi chạy thử Dashboard, một số chữ khó đọc: nhãn "FPS (CURRENT)", đơn vị
"fps", chữ "(Current)" cạnh "Detected Vehicles", và dòng "No data yet. Run
detection." đều gần như biến mất trên nền trắng. Đây không phải cảm tính —
đo được bằng công thức tương phản WCAG (không dùng thư viện, tự tính bằng
tay để hiểu rõ công thức):

```python
def lin(c):                              # tuyến tính hóa một kênh màu (0-255 -> 0-1)
    c = c / 255.0
    return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055) ** 2.4

def luminance(hexcolor):                 # độ sáng tương đối của cả màu (0 = đen, 1 = trắng)
    r, g, b = (int(hexcolor[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = lin(r), lin(g), lin(b)
    return 0.2126*r + 0.7152*g + 0.0722*b   # trọng số theo cảm nhận mắt người (giống §2.3 báo cáo)

def contrast(c1, c2):                    # tỉ lệ tương phản giữa 2 màu, luôn ≥ 1
    l1, l2 = luminance(c1), luminance(c2)
    l1, l2 = max(l1, l2), min(l1, l2)
    return (l1 + 0.05) / (l2 + 0.05)     # +0.05 tránh chia cho số quá nhỏ khi cả hai gần đen
```

Chuẩn WCAG AA yêu cầu **≥ 4.5:1** cho chữ thường, **≥ 3:1** cho chữ lớn
(≥18px hoặc ≥14px đậm). Chạy hàm `contrast()` cho từng cặp (màu chữ, màu
nền) trong `CUSTOM_CSS` phát hiện:

| Chữ | Màu cũ trên nền | Tỉ lệ đo được | Kết luận |
|---|---|---|---|
| "FPS (CURRENT)", đơn vị "fps" | `#94a3b8` trên `#ffffff` | **2,56:1** | Fail nặng — dưới nửa mức yêu cầu |
| "No data yet...", "(Current)" | `#94a3b8` trên `#ffffff` | **2,56:1** | Fail nặng |
| Nhãn "Vùng ứng viên" (thẻ xe) | `#64748b` trên `#eff6ff` | 4,37:1 | Fail sát nút (cần 4,5) |
| "Upload a video to start" | `#64748b` trên `#0f172a` (nền TỐI) | 3,75:1 | Fail — sai hướng, chữ tối trên nền tối |

### 3.8.1 Nguyên nhân gốc: theme chưa được ép cứng

Trước khi sửa màu CSS, có một nguyên nhân sâu hơn cần xử lý trước.
`.streamlit/config.toml` trước đây chỉ có:

```toml
[server]
maxUploadSize = 1000
```

Không có mục `[theme]`. Khi không khai báo, Streamlit tự chọn theme
**sáng hoặc tối theo hệ điều hành/trình duyệt của người xem** ("Auto").
Nhưng `CUSTOM_CSS` trong `app_streamlit.py` lại **ép cứng nền sáng** cho
nhiều khối (`[data-testid="stAppViewContainer"] > div:first-child {
background: #f0f2f6; }`), bất kể theme là gì. Hệ quả: nếu máy người xem
đang bật dark mode, mọi widget gốc của Streamlit mà CUSTOM_CSS không
với tới được — slider, checkbox, radio, number_input, `st.metric`, nút
tải CSV/JSON, bảng dataframe — sẽ tự động đổi sang chữ **màu sáng** (theo
theme tối), đặt trên nền sáng bị ép cứng phía trên → gần như vô hình.
Đây là lỗi tương phản có phạm vi rộng nhất, nhưng không xuất hiện trong
bảng đo ở trên vì bảng đó chỉ đo các đoạn có `color:` viết tay trong
`CUSTOM_CSS`, không đo được các widget do chính Streamlit tự vẽ.

**Sửa tận gốc** — ghi đè `.streamlit/config.toml` (dòng 13–18):

```toml
[theme]
base = "light"                              # dòng 14 — luôn theme sáng, bỏ qua theme của người xem
primaryColor = "#2563eb"                    # dòng 15 — trùng .fps-value, nút chính
backgroundColor = "#ffffff"                 # dòng 16 — nền các card/container
secondaryBackgroundColor = "#f0f2f6"        # dòng 17 — trùng nền trang đã ép ở CUSTOM_CSS
textColor = "#1e293b"                       # dòng 18 — trùng .section-title/.info-value
```

`base = "light"` là dòng quan trọng nhất: nó buộc mọi widget gốc luôn vẽ
chữ tối trên nền sáng, bất kể người xem đang dùng theme gì. Bốn dòng còn
lại chỉ đồng bộ màu chính thức với các màu đã dùng sẵn trong CUSTOM_CSS,
để giao diện không bị "vênh" giữa phần do Streamlit vẽ và phần do CSS vẽ.

**Lưu ý khi áp dụng:** thay đổi `[theme]` trong `config.toml` **không
tự nạp lại** khi Streamlit đang chạy (khác với sửa code Python, vốn có
nút "Rerun" tự động) — phải dừng hẳn tiến trình (`Ctrl+C` hoặc kill) rồi
`streamlit run app_streamlit.py` lại thì theme mới mới có hiệu lực.

### 3.8.2 Sửa các màu chữ tự viết trong CUSTOM_CSS

Sau khi ép theme, các đoạn `color:` viết tay trong CSS vẫn cần sửa riêng
vì chúng không phụ thuộc theme. Chọn một màu xám đậm hơn, `#52606d`, để
thay cho `#94a3b8` **và** `#64748b` ở mọi chỗ dùng trên nền sáng — trước
đó hai mã màu gần giống nhau này bị dùng lẫn lộn (`#64748b` chỗ này,
`#94a3b8` chỗ khác) dù cùng đóng vai trò "chữ phụ, mờ hơn chữ chính".
Gộp về một màu vừa sửa lỗi, vừa làm giao diện nhất quán hơn:

```
#52606d trên #ffffff : 6,46:1   (nhãn "FPS (CURRENT)", "No data yet", "(Current)")
#52606d trên #eff6ff : 5,93:1   (nhãn "Vùng ứng viên" trên thẻ xanh)
#52606d trên #f8fafc : 6,17:1   (tiêu đề cột bảng kết quả)
```

Danh sách 8 chỗ đã đổi từ `#64748b`/`#94a3b8` → `#52606d` (nền sáng, cần
**đậm** hơn):

```python
.vehicle-label  { ... color: #52606d; ... }   # dòng 100 — nhãn dưới icon xe trong thẻ đếm
.fps-label      { ... color: #52606d; ... }   # dòng 109 — chữ "FPS (CURRENT)"
.fps-unit       { ... color: #52606d; ... }   # dòng 111 — chữ "fps" cạnh số lớn
.info-label     { color: #52606d; ... }       # dòng 113 — nhãn "Video Name", "Duration"...
.results-table th { ... color: #52606d; ... } # dòng 126 — tiêu đề cột bảng kết quả
```
```python
return '<div style="... color: #52606d; ...">No data yet. Run detection.</div>'
                                                # dòng 142 — placeholder khi chưa có dữ liệu
st.markdown("<div style='... color: #52606d;'>OR</div>", ...)
                                                # dòng 232 — chữ "OR" giữa hai cách chọn video
st.markdown('<p class="section-title">Detected Vehicles '
            '<span style="color:#52606d;...">(Current)</span></p>', ...)
                                                # dòng 348 — chú thích nhỏ cạnh tiêu đề mục
```

Hai chỗ còn lại đi **ngược hướng** — nền của chúng là màu **tối**
(`.video-preview { background: #0f172a; }`), nên chữ phải **sáng hơn**
thay vì đậm hơn. Đổi `#64748b` (3,75:1, fail) → `#94a3b8` (6,96:1, đạt),
tái dùng đúng màu đã chứng minh đọc tốt trên nền tối ở `.header-subtitle`:

```python
video_placeholder.markdown('...<div style="color:#94a3b8; ...">Upload a video to start</div>...')
                                                # dòng 313 — chữ giữa khung video khi chưa chọn video
video_placeholder.markdown('...<div style="color:#94a3b8; ...">Ready to run detection</div>...')
                                                # dòng 315 — chữ giữa khung video khi đã chọn nhưng chưa Run
```

Những chỗ **không đổi** vì đã đủ tương phản từ đầu: `.header-subtitle`
(6,96:1, nền tối), `.badge-ready` xanh lá (7,04:1, nền tối), `.fps-value`
xanh dương đậm (5,17:1, nền trắng), `.info-value`/`.section-title` gần
đen (14,6:1), `.results-table td` (10,3:1), `.row-highlight td` (8,0:1).
Không sửa những chỗ này để tránh đổi giao diện nhiều hơn mức cần thiết.

### 3.8.3 Cách kiểm tra lại

Vì lỗi gốc chỉ lộ ra khi trình duyệt người xem ở dark mode, kiểm tra bằng
mắt trên máy mình (thường ở light mode) sẽ không phát hiện được. Dùng
Playwright ép `color_scheme="dark"` để mô phỏng đúng tình huống lỗi:

```python
page = browser.new_page(color_scheme="dark")   # giả lập trình duyệt của người dùng đang bật dark mode
page.goto("http://localhost:8501")
page.wait_for_selector("text=TransDetect-Vid")  # đợi Streamlit render xong (đừng dùng networkidle —
page.wait_for_timeout(2000)                     # Streamlit cập nhật qua WebSocket, không qua HTTP)
page.screenshot(path="dashboard_dark.png", full_page=True)
```

Sau khi ép `base = "light"`, ảnh chụp với `color_scheme="dark"` và
`color_scheme="light"` cho kết quả **giống hệt nhau** — đúng như mong đợi,
vì giờ theme không còn phụ thuộc vào lựa chọn của trình duyệt nữa.

---

## Phần 4 — `legacy/` và các script cũ

`legacy/README.md` nói rõ: bản sao lịch sử, giữ để script rất cũ dùng
`from preprocessing import preprocess` không bị vỡ. **Không có gì trong codebase
hiện tại import chúng.**

Đây là bản nháp gốc của nhóm — chú thích đúng nghĩa đen từng dòng, viết trước khi
đóng gói vào `src/transdetect/`. Đọc để hiểu giải thích thì tốt, nhưng **đừng
trích làm phần hiện thực**: chúng khác `src/` về thuật toán.

| | `legacy/` | `src/transdetect/` (báo cáo) |
|---|---|---|
| `epsilon` của ngưỡng | `0.5` | `1e-3` |
| Giá trị trả về của ngưỡng | `int(T)` | `(mask, float T)` |
| Phép hình thái | `MORPH_CLOSE` (giãn + mòn) | chỉ `dilate` |
| Chế độ contour | `RETR_EXTERNAL` | `RETR_LIST` |
| Định dạng box | `(x, y, w, h)` | `[x1, y1, x2, y2]` |
| Lọc diện tích | chỉ `min_area` | `min_area` **và** `max_area` |
| Lọc tỉ lệ khung | không có | hai phía, `0.25 ≤ w/h ≤ 4.0` |
| Optical flow | hàm rời, người gọi tự giữ trạng thái | class `LucasKanadeTracker` tự giữ |
| Ngưỡng Sobel mặc định | `50` | `40` |
| Có demo `__main__` | có (`cv2.imshow`) | không |

Các file ở thư mục **gốc** về bản chất cũng là legacy:

- **`yolo_detector.py` (gốc)** — hàm rời `load_model / detect_frame / draw_boxes`,
  trả **tuple** `(x1,y1,x2,y2,name,conf)` thay vì dict, và **không lọc lớp phương
  tiện** (báo cáo mọi lớp COCO, kể cả `person`, `traffic light`).
- **`app.py`** — giao diện Streamlit thứ hai, cũ hơn. Import `yolo_detector` gốc,
  hard-code `runs\detect\motorbike_yolo11n\weights\best.pt`, có tuỳ chọn
  `frame_skip` mà bản mới không có.
- **`run_demo.py`** — thử nhanh trên một ảnh, hard-code `D:\1ComputerVisionProject1\...`
  (đường dẫn riêng của một máy, nay không còn).
- **`train_yolo.py`** — `train()` gọi `model.train(...)` với `epochs`, `imgsz`,
  `batch=16`, `patience=15` (early stopping sau 15 epoch không cải thiện);
  `evaluate()` in `mAP50`, `mAP50-95`, `Precision`, `Recall`. Nhưng phần
  `__main__` **không có argparse** — bốn đường dẫn dataset bị hard-code tuyệt đối,
  trong đó có dataset biển báo giao thông nằm ngoài phạm vi đề tài.

---

## Phần 5 — Sơ đồ tổng

```
TRUYỀN THỐNG (main.py --method classical)
  _open_capture() → kiểm tra isOpened()  →  raise sớm nếu path sai
  VideoCapture.read()  →  frame BGR (H,W,3) uint8
    ├─ preprocess_frame ─→ cvtColor BGR2GRAY → equalizeHist → medianBlur(5)   ⇒ gray (H,W)
    │    └─ detect_vehicle_candidates
    │         ├─ iterative_global_threshold  ⇒ mask nhị phân + T   (2-means trên cường độ)
    │         ├─ sobel_edge_detection(40)    ⇒ mask biên           (‖∇I‖ rồi ngưỡng)
    │         ├─ bitwise_or → dilate(5×5)    ⇒ mask kết hợp        (hàn viền đứt đoạn)
    │         └─ findContours → lọc diện tích + tỉ lệ ⇒ [[x1,y1,x2,y2], …]
    │              └─ draw_classical_boxes   ⇒ khung chữ nhật xanh lá
    └─ cvtColor(THÔ) ─→ LucasKanadeTracker.track_features(prev_gray, curr_gray)
                          ├─ goodFeaturesToTrack (CHỈ khi tập điểm rỗng)
                          ├─ calcOpticalFlowPyrLK → lọc status==1
                          └─ v = p(t+1) − p(t)     ⇒ draw_motion_vectors (mũi tên đỏ)
                                                    → VideoWriter.write()

YOLO11 (main.py --method yolo)
  Yolo11VehicleDetector(...)  ← khởi tạo MỘT LẦN ngoài vòng lặp
  VideoCapture.read()  →  frame BGR
    └─ detect_frame(conf=.25, iou=.45, max_det=300)
         └─ model.predict  [letterbox 640² → Backbone → Neck FPN/PAN → Detect head
                            → giải mã DFL (8400 vị trí) → sigmoid → lọc conf → NMS]
              └─ lọc class_name ∈ VEHICLE_NAMES        (theo TÊN, không theo COCO ID)
                   ⇒ [{"bbox":[x1,y1,x2,y2], "class":…, "confidence":…}, …]
                        └─ draw_yolo_detections → VideoWriter.write()

DASHBOARD (app_streamlit.py) — chạy đúng hai nhánh trên (Classical CÓ Lucas-Kanade),
    KHÔNG đi qua pipeline.py, cộng thêm:
      · slider conf/iou/max_det  → truyền thẳng vào model.predict (dòng 473-475)
      · CLASS_ALIAS 6→4 + checkbox Target Classes  (dòng 481-487)
      · FPS(Current) cửa sổ trượt 10 frame | Average FPS lũy kế  (dòng 432-440)
      · throttle UI 30 ms + thu nhỏ ảnh về 480p  (dòng 524-540)
      · điều tiết phát 1.0× → FPS bị chặn trần bởi FPS nguồn  (dòng 579-583)
      · xuất CSV/JSON {frame_id, class_name, confidence, count}  (dòng 504-521)
```
