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
├── app_streamlit.py          # Web dashboard, dùng package transdetect
├── datasets/
│   ├── README.md             # Nguồn + giấy phép dataset (kèm link)
│   └── configs/
│       └── example_data.yaml
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

Pipeline được gom trong `src/transdetect/pipeline.py`. Ví dụ chạy trong Python:

```python
from transdetect import pipeline

# Nhánh truyền thống (Threshold + Sobel + Lucas-Kanade)
pipeline.run_classical("data/sample.mp4", "outputs/classical_out.mp4")

# Nhánh YOLO11
pipeline.run_yolo("data/sample.mp4", "outputs/yolo_out.mp4",
                  model_path="yolo11n.pt", conf=0.25, iou=0.45)
```

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

Báo cáo (Listing 3.4) dùng COCO ID `{2, 3, 5, 7}` (car, motorcycle, bus, truck).
Đây là chỉ số chuẩn trong bộ nhãn MS COCO:
<https://docs.ultralytics.com/datasets/detect/coco/>

Tuy nhiên model fine-tune trên dataset Roboflow có thứ tự lớp khác COCO (xem
`datasets/README.md`). Vì vậy `yolo_detector.py` lọc theo **tên lớp** trước
(`model.names`) và chỉ dùng COCO ID làm phương án dự phòng — cách này chạy đúng
cho cả model COCO pre-trained lẫn model tự huấn luyện. Thuộc tính `model.names`
được mô tả trong tài liệu Ultralytics Predict mode:
<https://docs.ultralytics.com/modes/predict/>

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

