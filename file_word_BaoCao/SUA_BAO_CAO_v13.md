# Danh sách sửa báo cáo v12 → v13

## Trạng thái

| | Nội dung | Trạng thái |
|---|---|---|
| **E1–E6** | 8 sửa đổi mô tả code | ✅ **ĐÃ ÁP DỤNG TỰ ĐỘNG** |
| **E7** | Danh mục hình ảnh / bảng biểu | ⬜ Phải làm tay trong Word (~2 phút) |
| **E8** | Chụp lại Hình 4.1–4.4 + số Bảng 4.3 | ⬜ Phải làm tay |
| **E9** | Trang bìa | ⬜ Tùy chọn — cần bạn quyết tháng nộp |

**File đã tạo:** `C:\Users\dzyuu\Downloads\CVFINAL\TransDetect-Vid_BaoCao_v13.docx`
**Bản gốc v12 giữ nguyên, không bị ghi đè.**

Đã xác minh: 11 bảng và 8 ảnh còn nguyên; font Times New Roman 13pt / line 1.5 / justify
không đổi; Chương 1, 2, 5 và Tài liệu tham khảo giống hệt v12; mọi câu tự khai trung thực
(*"Chưa đo"*, *"chưa phải số liệu đo"*, *"Hai video chưa có nhãn ground-truth"*) còn nguyên.

---

## E1. Listing 3.4 — Mục 3.5.1, trang 34

**Lý do:** code đã bổ sung tham số `max_det`. Phần còn lại của Listing giữ nguyên,
code trong repo đã được chỉnh về đúng dạng vòng lặp từng box như báo cáo in.

**Sửa dòng khai báo hàm:**

Cũ:
```
    def detect_frame(self, bgr_frame, conf_threshold=0.25, iou_threshold=0.45):
```
Mới:
```
    def detect_frame(self, bgr_frame, conf_threshold=0.25, iou_threshold=0.45,
                     max_det=300):
```

**Sửa lời gọi predict — thêm 1 dòng `max_det=max_det`:**

Cũ:
```
        results = self.model.predict(
            bgr_frame,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )
```
Mới:
```
        results = self.model.predict(
            bgr_frame,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            device=self.device,
            verbose=False,
        )
```

---

## E2. Mục 3.5.3, trang 36 — bullet VEHICLE_NAMES

Câu cũ:
> • **VEHICLE_NAMES:** source hiện lọc theo tên car, motorcycle/motorbike, bus, truck
> và container truck. Cách này hỗ trợ cả trọng số COCO và mô hình tùy biến, nhưng giao
> diện đếm cần ánh xạ rõ các tên bổ sung.

Câu mới:
> • **VEHICLE_NAMES:** source lọc theo tên car, motorcycle/motorbike, bus, truck và
> container truck, hỗ trợ cả trọng số COCO lẫn mô hình tùy biến. Vì giao diện chỉ có bốn
> ô đếm, `config.CLASS_ALIAS` ánh xạ các tên bổ sung về đúng nhóm hiển thị
> (motorbike → Motorcycle, container truck → Truck); nhãn nào không có trong bảng ánh xạ
> sẽ bị loại thay vì cộng nhầm sang lớp khác.

**Bổ sung bullet mới ngay dưới (vì `max_det` giờ đã có tác dụng):**
> • **max_det = 300:** giới hạn số box tối đa mỗi frame sau NMS, khớp mặc định của
> Ultralytics. Giá trị quá thấp sẽ cắt bớt phát hiện trong cảnh đông xe máy.

---

## E3. Mục 3.7, trang 37 — đoạn kết (QUAN TRỌNG NHẤT)

Đoạn cũ:
> Đối chiếu với phiên bản repo công khai cho thấy các module lõi preprocessing,
> classical_detector, optical_flow, yolo_detector và pipeline đã có mã hiện thực. Tuy nhiên
> Dashboard vẫn là prototype: max_det và các checkbox chọn lớp chưa được truyền đầy đủ vào
> suy luận; cột Avg Conf trong bảng giao diện còn khởi tạo 0,0; FPS chịu ảnh hưởng bởi nhịp
> phát video và cập nhật UI. Vì vậy ảnh Dashboard chỉ là minh họa chạy demo, không phải bằng
> chứng benchmark hay đánh giá độ chính xác.

Đoạn mới:
> Đối chiếu với phiên bản repo công khai cho thấy các module lõi preprocessing,
> classical_detector, optical_flow, yolo_detector và pipeline đã có mã hiện thực. Toàn bộ
> tham số trên Control Panel đã được nối vào luồng suy luận: `conf`, `iou` và `max_det`
> truyền thẳng xuống `model.predict`, bốn checkbox Target Classes lọc kết quả theo tên lớp
> đã chuẩn hóa, và cột Avg Conf hiển thị confidence trung bình thực của từng lớp trong
> frame. Hai chỉ số FPS được tách riêng: *FPS (Current)* đo trên cửa sổ 10 frame gần nhất,
> còn *Average FPS* bằng tổng số frame đã xử lý chia tổng thời gian chạy. Tuy vậy cả hai vẫn
> bao gồm thời gian đọc/ghi video, vẽ kết quả, cập nhật giao diện và cơ chế sleep bám FPS
> nguồn, nên ảnh Dashboard là minh họa luồng demo chứ chưa phải bằng chứng benchmark hay
> đánh giá độ chính xác.

**Cũng ở Mục 3.7, câu đầu:**

Câu cũ:
> Chức năng xuất CSV/JSON chỉ được xem là hoàn chỉnh sau khi file tải xuống chứa dữ liệu
> thật và đã được kiểm tra.

Câu mới:
> Chức năng xuất CSV/JSON ghi lại frame_id, tên lớp, confidence trung bình và số lượng của
> từng lớp trên mỗi frame; với nhánh truyền thống, trường class_name là `candidate` và
> confidence để `N/A` do phương pháp này không phân loại loại xe.

---

## E4. Mục 3.3, trang 31 — bộ lọc tỉ lệ khung

Câu cũ:
> sau đó contour quá nhỏ hoặc quá lớn (lọc theo min_area và max_area) cùng contour có tỉ lệ
> khung sai đều bị loại để giảm false positive.

Câu mới:
> sau đó contour quá nhỏ hoặc quá lớn (lọc theo min_area và max_area) cùng contour có tỉ lệ
> khung sai đều bị loại để giảm false positive. Tỉ lệ w/h bị chặn ở **cả hai phía**, tức chỉ
> giữ contour thỏa 1/4 ≤ w/h ≤ 4: ngưỡng trên loại các vùng quá dẹt như vạch kẻ đường hay
> dải phân cách, ngưỡng dưới loại các vùng quá cao và hẹp như cột điện, thân cây.

---

## E5. Mục 4.2.1, trang 39 — làm rõ hai chỉ số FPS

Chèn thêm **sau** câu *"...không phải FPS suy luận thuần của thuật toán."*:

> Từ phiên bản hiện tại, Dashboard tách hai chỉ số: *FPS (Current)* đo trên cửa sổ 10 frame
> gần nhất để phản ánh tốc độ tức thời, còn *Average FPS* tính bằng tổng số frame đã xử lý
> chia tổng thời gian chạy kể từ lúc bấm Run. Trước đây hai ô cùng hiển thị một giá trị, nên
> các số trong Hình 4.1–4.4 của bản trước phải đọc là FPS tức thời.

---

## E6. Phụ lục A.1, trang 47 — cấu trúc thư mục

Bổ sung các dòng còn thiếu so với repo thật (giữ nguyên phần đã có):

```
TransDetect-Vid/
├── requirements.txt
├── README.md
├── main.py                  # Điểm chạy chính qua giao diện dòng lệnh (CLI)
├── app_streamlit.py         # Web dashboard hiển thị kết quả cho người dùng
├── train_yolo.py            # Huấn luyện / đánh giá YOLO11 trên dataset tùy chọn
├── .streamlit/
│   └── config.toml
├── test_videos/
├── outputs/
├── datasets/
│   └── README.md            # Nguồn Roboflow + giấy phép; KHÔNG commit ảnh/nhãn
├── notebooks/
│   └── TransDetect_Vid_Colab_Demo.ipynb
└── src/
    └── transdetect/
        ├── __init__.py
        ├── config.py             # Tham số tập trung + CLASS_ALIAS
        ├── preprocessing.py
        ├── classical_detector.py # Cài đặt Thresholding và Sobel
        ├── optical_flow.py       # Theo dõi chuyển động Lucas-Kanade
        ├── yolo_detector.py      # Cài đặt mô hình học sâu YOLO11
        ├── visualization.py
        └── pipeline.py
```

> ⚠️ Repo hiện còn `app.py`, `yolo_detector.py` (ở thư mục gốc), `run_demo.py` và `legacy/`
> — đều là bản cũ trùng chức năng với `src/transdetect/`, cùng `datasets/` chứa 4.550 file
> nhãn (trái với chính `datasets/README.md`). **Nên xóa trước khi nộp** thì cây thư mục trên
> mới đúng. Nếu giữ lại thì phải liệt kê đủ trong Phụ lục A.1.

**Phụ lục A.2, trang 47** — bổ sung ví dụ `--max-det`:
```
python main.py --input test_videos/sample.mp4 --method yolo --model yolo11n.pt ^
               --conf 0.25 --iou 0.45 --max-det 300 --output outputs/yolo_out.mp4
```

---

## E7. Trang 7 & 8 — Danh mục hình ảnh / bảng biểu ĐANG RỖNG

Hiện cả hai trang đều in `No table of contents entries found.` — vi phạm trực tiếp yêu cầu
*"Danh mục hình ảnh (liên kết tự động đến trang)"* trong hướng dẫn trình bày.

> **⚠️ ĐÍNH CHÍNH so với bản đầu của tài liệu này.** Ban đầu tôi hướng dẫn phải làm lại
> toàn bộ caption bằng Insert Caption. **Không cần.** Kiểm tra file .docx cho thấy 17 caption
> đã được gán sẵn hai style riêng là **`Caption Hinh`** và **`Caption Bang`** — chỉ thiếu
> field SEQ. Word cho phép sinh danh mục **theo style**, nên việc này chỉ mất ~2 phút.

**Cách sửa (2 phút):**
1. Đặt con trỏ vào trang 7 (dưới tiêu đề *DANH MỤC HÌNH ẢNH*), xóa dòng
   `No table of contents entries found.`
2. **References → Insert Table of Figures → Options…**
3. Bỏ tick *Style* mặc định, tick **Style** rồi chọn **`Caption Hinh`** → OK → OK.
4. Sang trang 8, lặp lại bước 2–3 nhưng chọn **`Caption Bang`**.
5. Cuối cùng: Ctrl+A → F9 → *Update entire table* để cập nhật cả Mục lục.

**Sẽ tự động gom đủ:** Hình 2.1, 3.1, 3.2, 4.1, 4.2, 4.3, 4.4 (7 hình) và
Bảng 2.1, 2.2, 2.3, 3.1, 3.2, 4.1, 4.2, 4.3, 4.4, 5.1 (10 bảng).

---

## E8. Chương 4 — PHẢI chụp lại Hình 4.1 → 4.4

Bắt buộc, vì:
- Caption Hình 4.3 đang tự khai *"của Dashboard bản trước khi sửa"*.
- Hình 4.1 hiển thị số **72** nằm dưới nhãn **"Car"**, trong khi code hiện tại render
  nhãn **"Vùng ứng viên"**.
- Ô *Average FPS* trong ảnh cũ trùng khít ô *FPS (Current)* — nay đã là hai giá trị khác nhau.
- Cột *Avg Conf* trong bảng nay có số thật thay vì 0.00.

**Chụp cả 4 ảnh trên CÙNG một máy** rồi ghi rõ CPU/GPU vào Bảng 4.2 — hiện Bảng 4.3 đang
so CPU laptop với RTX 3060/Colab T4 nên không kết luận được gì.

---

## E9. Trang bìa — đối chiếu mẫu hướng dẫn

| Mẫu hướng dẫn | Báo cáo hiện tại | Ghi chú |
|---|---|---|
| `ĐỒ ÁN MÔN HỌC` | `BÁO CÁO KẾT THÚC HỌC PHẦN` | Nên đổi cho khớp mẫu |
| `TP. HỒ CHÍ MINH, 8/2026` | `Thành phố Hồ Chí Minh, tháng 6 năm 2026` | Chỉnh theo tháng nộp thật |

Kiểm tra lại định dạng toàn văn: **Times New Roman 13pt, line spacing 1.5, canh đều
(justify)**; mỗi chương và mỗi mục lớn bắt đầu ở **đầu trang mới**; Tài liệu tham khảo cũng
bắt đầu từ đầu trang mới.

---

## Những chỗ KHÔNG được sửa

| Nội dung | Lý do giữ nguyên |
|---|---|
| Bảng 4.3 cột chất lượng ghi *"Chưa đo"* | Trung thực. Chưa có ground-truth thì không được điền Precision/Recall/mAP. |
| Mục 4.2.2 *"không được suy diễn Precision/Recall/mAP từ ảnh minh họa"* | Đúng phương pháp luận. |
| Câu *"Đây là kết quả của công trình tham khảo, không phải kết quả của nhóm"* (tr.15) | Phân định rõ, tránh bị hiểu là đạo kết quả. |
| Bảng 4.4 ghi *"kỳ vọng định tính... chưa phải số liệu đo"* | Minh bạch. |
| Toàn bộ Chương 2 | Đã kiểm tra lại từng công thức và số ví dụ — đều đúng. |
| Listing 3.1, 3.2, 3.3 | Khớp chính xác với code. |
