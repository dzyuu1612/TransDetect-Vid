# 🏍️ Motorbike & Car Detector (YOLO11 & Classical CV)

Dự án này là một hệ thống phát hiện phương tiện giao thông (ô tô, xe máy) trong video và hình ảnh. Dự án cung cấp hai phương pháp tiếp cận:
1. **Phương pháp Deep Learning (YOLO11):** Sử dụng mô hình YOLO11 mới nhất để nhận diện độ chính xác cao.
2. **Phương pháp Truyền thống (Classical Computer Vision):** Sử dụng các kỹ thuật xử lý ảnh thủ công (Thresholding, Sobel, Optical Flow) để nhận diện và theo dõi vật thể chuyển động.

Dự án cung cấp cả giao diện dòng lệnh (CLI) và giao diện Web (Streamlit).

---

## 📂 Cấu Trúc Mã Nguồn (Source Code)

- **`app.py`**: Giao diện Web UI bằng Streamlit. Cho phép người dùng tải lên hình ảnh hoặc video để nhận diện bằng mô hình YOLO trực quan trên trình duyệt.
- **`main.py`**: File chạy chính trên giao diện dòng lệnh (CLI). Cho phép chọn chạy giữa pipeline truyền thống (`classical`) hoặc `yolo`.
- **`train_yolo.py`**: Script dùng để huấn luyện (train) và đánh giá (evaluate) mô hình YOLO11 trên dataset tùy chỉnh của bạn.
- **`yolo_detector.py`**: Các hàm tiện ích hỗ trợ nhận diện bằng YOLO (tải mô hình, phát hiện trên frame, vẽ bounding box).
- **`classical_detector.py`**: Pipeline phát hiện truyền thống, sử dụng phương pháp ngưỡng (thresholding) lặp tự động kết hợp với đạo hàm Sobel để tìm ứng viên vùng (candidate boxes).
- **`preprocessing.py`**: Chứa các bước tiền xử lý ảnh cho phương pháp truyền thống (chuyển ảnh xám, cân bằng sáng Histogram, lọc nhiễu Gaussian).
- **`optical_flow.py`**: Ứng dụng thuật toán Lucas-Kanade (Optical Flow) để theo dõi các điểm chuyển động liên tục giữa các frame (cho phương pháp truyền thống).
- **`run_demo.py`**: Một script nhỏ để chạy thử mô hình YOLO trực tiếp trên 1 ảnh kiểm thử và lưu kết quả.

---

## 🚀 Hướng Dẫn Cài Đặt

Trước khi chạy, hãy đảm bảo bạn đã cài đặt Python. Sau đó, cài đặt các thư viện cần thiết bằng lệnh sau:

```bash
pip install numpy opencv-python ultralytics streamlit
```

---

## 🖥️ Hướng Dẫn Sử Dụng

### 1. Giao Diện Web Trực Quan (Streamlit)
Cách nhanh nhất và trực quan nhất để thử nghiệm là dùng Web UI. Giao diện hỗ trợ tải lên cả Video (`.mp4`, `.avi`, `.mov`) và Ảnh (`.jpg`, `.png`).

```bash
streamlit run app.py
```
> **Lưu ý:** Lệnh này sẽ mở ra một trang web trên trình duyệt. Tại đây bạn có thể kéo thả file vào để nhận diện và điều chỉnh độ tin cậy (Confidence) ngay trên thanh trượt.

### 2. Sử Dụng Giao Diện Dòng Lệnh (CLI)
Bạn có thể chạy trực tiếp video qua command line với file `main.py`.

**Chạy với YOLO:**
```bash
python main.py --input path_to_video.mp4 --output result.mp4 --method yolo --model runs/detect/motorbike_yolo11n/weights/best.pt
```

**Chạy với Phương Pháp Truyền Thống (Classical):**
```bash
python main.py --input path_to_video.mp4 --output result.mp4 --method classical
```

### 3. Huấn Luyện Mô Hình (Training)
Nếu bạn có tập dữ liệu mới và muốn huấn luyện lại YOLO11:
1. Chuẩn bị tập dữ liệu (có file `data.yaml`).
2. Sửa đường dẫn dataset trong phần `if __name__ == "__main__":` của file `train_yolo.py`.
3. Chạy lệnh:
```bash
python train_yolo.py
```
Mô hình tốt nhất sau khi huấn luyện sẽ được lưu tự động tại: `runs/detect/<tên_run>/weights/best.pt`.

---

## 📌 Lưu Ý Thêm
- **YOLO11** được tối ưu cực kỳ tốt, trong dự án sử dụng bản `yolo11n.pt` (Nano) rất nhẹ, phù hợp chạy ngay cả trên CPU thông thường.
- Phương pháp **Classical** được giữ lại trong mã nguồn với mục đích giáo dục, giúp hiểu rõ bản chất của xử lý ảnh, tuy nhiên độ chính xác và tính ổn định không thể so sánh với YOLO.
