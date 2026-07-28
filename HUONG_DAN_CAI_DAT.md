# HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY TRANSDETECT-VID

Tài liệu này mô tả cấu trúc package `src/transdetect/` đã được bổ sung để khớp
với Phụ lục A và các Listing trong báo cáo. Cấu trúc cũ (các file ở thư mục gốc
như `main.py`, `app.py`, `classical_detector.py`...) **vẫn được giữ nguyên** để
không phá vỡ các tham chiếu cũ; phần dưới đây hướng dẫn dùng package mới.

Repository: <https://github.com/dzyuu1612/TransDetect-Vid>

---

## 1. Cấu trúc package mới (khớp Phụ lục A)

```text
TransDetect-Vid/
├── requirements.txt
├── pyproject.toml
├── main.py                   # CLI, gọi src/transdetect/pipeline.py
├── app_streamlit.py          # Web dashboard, dùng package transdetect
├── train_yolo.py             # Huấn luyện / đánh giá YOLO11
├── CODE_WALKTHROUGH.md       # Giải thích mã nguồn từng dòng
├── datasets/
│   ├── README.md             # Nguồn + giấy phép dataset (kèm link)
│   └── configs/
│       └── example_data.yaml
├── notebooks/
│   └── TransDetect_Vid_Colab_Demo.ipynb
└── src/
    └── transdetect/
        ├── __init__.py
        ├── config.py
        ├── preprocessing.py       # Listing 3.1
        ├── classical_detector.py  # Listing 3.2
        ├── optical_flow.py        # Listing 3.3 (LucasKanadeTracker)
        ├── yolo_detector.py       # Listing 3.4 (Yolo11VehicleDetector)
        ├── visualization.py
        └── pipeline.py
```

Bố cục package `src/` theo chuẩn "src layout" của Python Packaging Authority:
<https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>

### File cũ còn nằm trong repo (KHÔNG dùng cho báo cáo)

Những file dưới đây là bản trước khi refactor, giữ lại để không phá vỡ tham
chiếu cũ. Chúng **không** được import bởi `main.py` hay `app_streamlit.py`, và
**khác thuật toán** so với `src/transdetect/`. Chi tiết khác biệt xem bảng so
sánh ở Mục 4 của `CODE_WALKTHROUGH.md`.

| File | Là gì | Vì sao đừng trích vào báo cáo |
|---|---|---|
| `legacy/preprocessing.py` | bản nháp chú thích từng dòng | trùng chức năng với `src/transdetect/preprocessing.py` |
| `legacy/classical_detector.py` | bản nháp | dùng `MORPH_CLOSE` + `RETR_EXTERNAL`, box `(x,y,w,h)`, không lọc `max_area`/tỉ lệ khung |
| `legacy/optical_flow.py` | bản nháp | hàm rời, người gọi tự giữ trạng thái điểm |
| `yolo_detector.py` (gốc) | API cũ | trả tuple thay vì dict, **không lọc lớp phương tiện** |
| `app.py` | dashboard Streamlit cũ | hard-code `runs\detect\motorbike_yolo11n\weights\best.pt` |
| `run_demo.py` | thử nhanh 1 ảnh | hard-code đường dẫn máy cá nhân, không chạy được |

---

## 2. Cài đặt

```bash
git clone https://github.com/dzyuu1612/TransDetect-Vid.git
cd TransDetect-Vid
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -e .
```

Lệnh `pip install -e .` cài repo ở chế độ phát triển (editable install) theo
pyproject.toml: <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>

---

## 3. Chạy bằng package transdetect

Pipeline được gom trong `src/transdetect/pipeline.py`.

### Chạy bằng CLI (cách trong Phụ lục A.2)

```bash
# Nhánh truyền thống — video kết quả có box ứng viên VÀ mũi tên Lucas-Kanade
python main.py --input test_videos/sg2.mp4 --method classical \
    --output outputs/classical_out.mp4

# Nhánh YOLO11
python main.py --input test_videos/sg2.mp4 --method yolo \
    --model yolo11n.pt --conf 0.25 --iou 0.45 --max-det 300 \
    --output outputs/yolo_out.mp4
```

Bỏ trống `--conf/--iou/--max-det` thì giá trị rơi về `src/transdetect/config.py`,
nên chỉ có một nguồn chân lý duy nhất cho tham số mặc định.

Đường dẫn video sai sẽ raise `FileNotFoundError` ngay lập tức, không còn âm thầm
ghi ra file rỗng.

### Hoặc gọi trực tiếp trong Python

```python
from transdetect import pipeline

pipeline.run_classical("test_videos/sg2.mp4", "outputs/classical_out.mp4")
pipeline.run_yolo("test_videos/sg2.mp4", "outputs/yolo_out.mp4",
                  model_path="yolo11n.pt", conf=0.25, iou=0.45)
```

> Thư mục `test_videos/` và `outputs/` không có trong repo (`.gitignore` loại
> media). Tự tạo `test_videos/` và bỏ video vào; `outputs/` được pipeline tự tạo.

### Streamlit

```bash
streamlit run app_streamlit.py
```

Tài liệu Streamlit: <https://docs.streamlit.io/get-started>

---

## 4. Đối chiếu Listing trong báo cáo với mã nguồn

| Listing báo cáo | File trong repo | Khớp |
|---|---|---|
| Listing 3.1 (preprocessing) | `src/transdetect/preprocessing.py` | `histogram_equalization`, `median_filter`, `preprocess_frame` |
| Listing 3.2 (classical) | `src/transdetect/classical_detector.py` | trả box `xyxy`, lọc `max_aspect_ratio` |
| Listing 3.3 (optical flow) | `src/transdetect/optical_flow.py` | lớp `LucasKanadeTracker` |
| Listing 3.4 (YOLO11) | `src/transdetect/yolo_detector.py` | lớp `Yolo11VehicleDetector`, `conf` + `iou` |

### Ghi chú về lọc lớp YOLO11

`yolo_detector.py` lọc **hoàn toàn theo tên lớp**, không dùng COCO ID ở bất kỳ
đâu — kể cả làm phương án dự phòng. Điều này khớp Mục 3.5 của báo cáo v14: "Các
COCO ID {2, 3, 5, 7} chỉ mô tả ánh xạ của trọng số chuẩn, không phải điều kiện
lọc đang dùng trong source."

Lý do: model fine-tune trên dataset Roboflow có thứ tự lớp khác COCO (xem
`datasets/README.md`), nên lọc theo ID sẽ lấy nhầm lớp. Lọc theo tên chạy đúng
cho cả trọng số COCO pre-trained lẫn model tự huấn luyện.

Cơ chế gồm hai bảng, đặt ở hai tầng khác nhau:

| Bảng | Ở đâu | Vai trò |
|---|---|---|
| `VEHICLE_NAMES` (6 tên) | `src/transdetect/yolo_detector.py` | Tầng phát hiện: giữ box nếu `model.names[class_id]` (đã `.strip().lower()`) nằm trong tập này |
| `CLASS_ALIAS` (6 → 4) | `src/transdetect/config.py` | Tầng hiển thị: gộp `motorbike → Motorcycle`, `container truck → Truck` cho khớp 4 ô đếm trên UI |

Tên nào không có trong `CLASS_ALIAS` sẽ bị **loại bỏ**, không bao giờ cộng nhầm
sang lớp khác. Thuộc tính `model.names` được mô tả trong tài liệu Ultralytics
Predict mode: <https://docs.ultralytics.com/modes/predict/>

---

## 5. Cơ sở lý thuyết và tài liệu nguồn

| Kỹ thuật | Hàm OpenCV/Ultralytics | Tài liệu nguồn |
|---|---|---|
| Histogram Equalization | `cv2.equalizeHist` | <https://docs.opencv.org/4.x/d4/d1b/tutorial_histogram_equalization.html> |
| Median Filter | `cv2.medianBlur` | <https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html> |
| Global Thresholding | `cv2.threshold` | <https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html> |
| Sobel Edge | `cv2.Sobel` | <https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html> |
| Shi-Tomasi corners | `cv2.goodFeaturesToTrack` | <https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html> |
| Lucas-Kanade Optical Flow | `cv2.calcOpticalFlowPyrLK` | <https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html> |
| YOLO11 | `ultralytics.YOLO` | <https://docs.ultralytics.com/models/yolo11/> |

Bài báo gốc:

- Lucas & Kanade (1981): <https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_2/lucas_bruce_d_1981_2.pdf>
- Bouguet, Pyramidal LK (2001): <http://robots.stanford.edu/cs223b04/algo_tracking.pdf>
- Redmon et al., YOLO (2016): <https://arxiv.org/abs/1506.02640>

---

## 6. Phạm vi đề tài

Đề tài chỉ đánh giá **phương tiện giao thông** (xe máy, ô tô, xe buýt, xe tải),
đúng như Mục 1.3 của báo cáo. Khi huấn luyện lại YOLO11, chỉ dùng các dataset
phương tiện liệt kê trong `datasets/README.md`; không đưa dataset biển báo giao
thông vào phạm vi đánh giá.

---

## 7. Lưu ý bảo mật

Không commit token, API key hay đường dẫn máy cá nhân vào repo. Nếu lỡ commit
secret, hãy thu hồi (revoke) ngay tại nơi cấp. Tham khảo hướng dẫn xử lý secret
bị lộ của GitHub:
<https://docs.github.com/en/code-security/secret-scanning/about-secret-scanning>

---

## 8. Các điểm chưa khớp còn tồn đọng

Ghi lại để không quên, xếp theo mức ưu tiên:

1. **`train_yolo.py` chưa có argparse.** `datasets/README.md` hướng dẫn
   `python train_yolo.py --data ... --model-size n --epochs 50`, nhưng file thực
   tế hard-code 4 đường dẫn tuyệt đối `D:\1ComputerVisionProject1\...` trong
   `__main__` — mâu thuẫn với chính Mục 7 ở trên. Danh sách đó còn chứa dataset
   biển báo giao thông, thứ mà Mục 6 tuyên bố nằm ngoài phạm vi.
2. **Nhãn dataset đang bị commit.** `.gitignore` chặn ảnh (0 file ảnh trong
   repo) nhưng không chặn `.txt`, nên 4.534 file nhãn vẫn nằm trong lịch sử Git,
   trái với ghi chú "KHÔNG commit ảnh/nhãn" ở Phụ lục A.1 của báo cáo.
3. **Cần đo lại FPS ở Bảng 4.3.** Dashboard vừa được bổ sung Lucas-Kanade cho
   nhánh Classical (mỗi frame chạy thêm `calcOpticalFlowPyrLK`, và
   `goodFeaturesToTrack` mỗi khi tập điểm cạn), nên khoảng 6,4–14,6 FPS trong
   báo cáo không còn phản ánh code hiện tại.
4. **`README.md` là bản trước refactor**, còn ghi `classical_detector.py`,
   `preprocessing.py`, `optical_flow.py` nằm ở thư mục gốc.

