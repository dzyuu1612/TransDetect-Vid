# Datasets

Các dataset huấn luyện không nên được commit trực tiếp vào Git để tránh
repository quá lớn và lỗi đường dẫn dài trên Windows. Thư mục này chỉ giữ tài
liệu mô tả nguồn và file cấu hình mẫu.

## Nguồn dữ liệu

Các dataset dưới đây lấy từ Roboflow Universe. Đường dẫn và giấy phép được
trích trực tiếp từ file `data.yaml` của từng dataset.

| Dataset | Phiên bản | Lớp | Giấy phép | Nguồn |
|---|---:|---|---|---|
| Car Motorcycle Truck | v1 | car, motorcycle, truck | CC BY 4.0 | https://universe.roboflow.com/chanyanuch/car-motorcycle-truck/dataset/1 |
| Vehicle Detection | v7 | bus, car, motorcycle, truck | CC BY 4.0 | https://universe.roboflow.com/isabinimam/vehicle-detection-ashiv/dataset/7 |
| Vietnam Container truck | v1 | Container truck | CC BY 4.0 | https://universe.roboflow.com/anh-khoa/vietnam-container-truck/dataset/1 |

> Phạm vi đề tài chỉ đánh giá phương tiện giao thông (xe máy, ô tô, xe buýt,
> xe tải). Dataset biển báo giao thông không thuộc phạm vi đề tài này.

## Cấu trúc sau khi tải

```text
datasets/local/vehicle/
├── data.yaml
├── train/
├── valid/
└── test/
```

## Tải dữ liệu

Tải dataset từ các URL Roboflow ở trên (định dạng YOLOv11), giải nén vào
`datasets/local/`, rồi trỏ đường dẫn `data.yaml` qua tham số `--data` khi train:

```bash
python train_yolo.py --data datasets/local/vehicle/data.yaml --model-size n --epochs 50
```

Xem `datasets/configs/example_data.yaml` để biết định dạng cấu hình mẫu.

> Không commit ảnh/nhãn dataset, API key Roboflow hoặc URL riêng tư vào repo.
