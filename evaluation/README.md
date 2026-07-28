# Tập đánh giá định lượng (ground truth)

Thư mục này chứa dữ liệu để chạy `evaluate_pipelines.py` — so sánh nhánh
truyền thống và YOLO11 bằng Precision / Recall / F1 tại IoU ≥ 0,5.

**Trạng thái hiện tại: CHƯA CÓ DỮ LIỆU.** Ba thư mục con đang rỗng (chỉ có
`.gitkeep`). Vì vậy các ô số liệu trong Chương 4 của báo cáo vẫn để dạng
placeholder `[N]`, `[TP_C]`, `[F1_C]`… và chưa được điền.

## Cấu trúc

```text
evaluation/
├── images/     # Frame trích từ video, ví dụ frame_000001.jpg
├── labels/     # Nhãn YOLO cùng tên, ví dụ frame_000001.txt
└── results/    # CSV do script sinh ra (không gán nhãn tay ở đây)
```

Mỗi ảnh phải có **đúng một** file nhãn cùng tên (khác phần mở rộng). Ảnh
thiếu nhãn sẽ bị coi như không có phương tiện nào, làm Precision giảm sai
lệch — script sẽ in cảnh báo nếu số ảnh và số nhãn không khớp.

## Định dạng nhãn

Định dạng YOLO, mỗi dòng một phương tiện, toạ độ chuẩn hoá về `[0, 1]`:

```text
class_id x_center y_center width height
```

`class_id` **bị bỏ qua** khi đánh giá, vì phép so sánh là *class-agnostic*
(gộp Car/Motorcycle/Bus/Truck thành một lớp `vehicle`). Lý do: nhánh truyền
thống chỉ sinh vùng ứng viên, không phân loại được loại xe — nếu so sánh có
phân biệt lớp thì nhánh này sẽ sai 100%, không phản ánh đúng khả năng **định
vị** phương tiện của nó. Dù vậy vẫn nên gán `class_id` đúng để sau này có
thể đánh giá class-aware riêng cho YOLO11.

## Cách chuẩn bị

1. **Trích frame phân bố đều** trên toàn video (ví dụ video 780 frame, lấy
   100 frame thì bước ≈ 8). Không lấy 100 frame liên tiếp vì các frame gần
   nhau gần như giống hệt nhau. Không chọn frame dựa trên việc YOLO chạy đẹp
   hay xấu — sẽ gây thiên lệch.
2. **Gán nhãn thủ công** bằng LabelImg, CVAT hoặc Roboflow Annotate. Bao sát
   phần nhìn thấy của phương tiện.
3. **Không dùng output của YOLO làm ground truth.** Nếu lấy dự đoán của YOLO
   làm chuẩn rồi chấm điểm chính YOLO, kết quả sẽ thành vòng lặp tự khẳng
   định và vô nghĩa.
4. **Không bỏ qua một chiếc xe** chỉ vì nhánh truyền thống không phát hiện
   được nó.
5. Ghi lại **công cụ gán nhãn và người gán nhãn** để đưa vào Mục 4.1 của báo
   cáo.

## Chạy đánh giá

```bash
python evaluate_pipelines.py --images evaluation/images --labels evaluation/labels --model yolo11n.pt --conf 0.25 --iou-match 0.5 --output evaluation/results
```

Kiểm thử các hàm tính toán (không cần dữ liệu):

```bash
python evaluate_pipelines.py --selftest
```

## Lưu ý về dung lượng

`.gitignore` của repo đang loại trừ `*.jpg`, `*.png` nên ảnh trong `images/`
sẽ **không** được commit. Nếu cần nộp kèm bằng chứng, có thể commit riêng
20–30 frame minh chứng (dùng `git add -f`), còn toàn bộ tập đánh giá lưu ở
Google Drive/Roboflow và ghi URL cùng phiên bản dataset vào đây.
