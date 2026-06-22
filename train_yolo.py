"""
train_yolo.py
---------------
Huan luyen YOLO11 - giai thich CHI TIET TUNG DONG.
"""

from ultralytics import YOLO   # thu vien chinh thuc cua YOLO11, chua san model + ham train/val/predict


def train(data_yaml_path: str, model_size: str = "n", epochs: int = 50,
          img_size: int = 640, run_name: str = "transdetect_run"):
    """Huan luyen YOLO11 tren dataset cua ban."""
    # YOLO("yolo11n.pt"): tai model YOLO11 ban "n" (nano) DA duoc pre-train tren COCO
    # neu chua co file nay tren may, Ultralytics se TU DONG tai tu internet ve
    # f"yolo11{model_size}.pt" -> vi du model_size="n" se ra chuoi "yolo11n.pt"
    model = YOLO(f"yolo11{model_size}.pt")

    model.train(
        data=data_yaml_path,   # duong dan file data.yaml: noi ghi train/val o dau, co bao nhieu class, ten class gi
        epochs=epochs,          # so vong huan luyen: model se "nhin" qua toan bo du lieu train epochs lan
        imgsz=img_size,         # anh se duoc resize ve kich thuoc img_size x img_size truoc khi dua vao model
        batch=16,                # so anh xu ly cung luc trong 1 lan cap nhat trong so (giam neu het RAM/VRAM)
        name=run_name,           # ten thu muc luu ket qua, se nam trong runs/detect/<run_name>
        patience=15,             # neu sau 15 epoch lien tiep khong cai thien -> tu dung som (early stopping)
        plots=True,              # tu ve bieu do loss/precision/recall/mAP sau khi train xong
    )

    print(f"Huan luyen xong. Model tot nhat nam tai: runs/detect/{run_name}/weights/best.pt")


def evaluate(model_path: str, data_yaml_path: str):
    """Danh gia model da train tren tap validation."""
    model = YOLO(model_path)             # tai lai model da train (file .pt)
    metrics = model.val(data=data_yaml_path)   # chay danh gia tren tap validation trong data.yaml

    # metrics.box.* chua cac chi so danh gia chuan cho bai toan detection:
    print("mAP50:", metrics.box.map50)   # do chinh xac trung binh khi IoU >= 0.5 (de hon, thuong cao hon)
    print("mAP50-95:", metrics.box.map)  # do chinh xac trung binh tinh tren nhieu muc IoU 0.5->0.95 (khat khe hon)
    print("Precision:", metrics.box.mp)  # ty le du doan dung / tong so du doan (du doan it sai)
    print("Recall:", metrics.box.mr)     # ty le phat hien dung / tong so vat the thuc te (khong bo sot)


if __name__ == "__main__":
    datasets_to_train = [
        (r"D:\1ComputerVisionProject1\Vietnam-Traffic-Sign.v1i.yolov11\data.yaml", "traffic_sign_yolo11n"),
        (r"D:\1ComputerVisionProject1\Car Motorcycle Truck.v1i.yolov11\data.yaml", "car_motorcycle_truck_yolo11n"),
        (r"D:\1ComputerVisionProject1\Vietnam Container truck.v1i.yolov11\data.yaml", "vietnam_container_truck_yolo11n"),
        (r"D:\1ComputerVisionProject1\Vehicle Detection.v7i.yolov11\data.yaml", "vehicle_detection_yolo11n")
    ]

    for data_yaml, run_name in datasets_to_train:
        print(f"\n=======================================================")
        print(f"BAT DAU HUAN LUYEN: {run_name}")
        print(f"=======================================================\n")
        try:
            train(
                data_yaml_path=data_yaml,
                model_size="n",
                epochs=50,
                run_name=run_name
            )
        except Exception as e:
            print(f"Loi khi huan luyen {run_name}: {e}")


    # Bo comment de danh gia model sau khi train xong
    # evaluate(
    #     model_path="runs/detect/motorbike_yolo11n/weights/best.pt",
    #     data_yaml_path="datasets/motorbike/data.yaml"
    # )
