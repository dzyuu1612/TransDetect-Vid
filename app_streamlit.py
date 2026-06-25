"""
app_streamlit.py
------------------
Web dashboard Streamlit cho TransDetect-Vid.

Khớp tên file trong Phụ lục A của báo cáo. Dùng package src/transdetect.
Chạy: streamlit run app_streamlit.py
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from transdetect import config
from transdetect.yolo_detector import Yolo11VehicleDetector
from transdetect import visualization

st.set_page_config(page_title="TransDetect-Vid", page_icon="🚗")


@st.cache_resource
def load_detector(model_path: str):
    """Tải detector một lần và cache cho các lần chạy sau."""
    return Yolo11VehicleDetector(model_path)


def _resolve_model_path() -> str:
    """Ưu tiên model đã fine-tune nếu có, nếu không dùng model mặc định."""
    trained = os.path.join(
        "runs", "detect", "vehicle_yolo11n", "weights", "best.pt"
    )
    if os.path.exists(trained):
        return trained
    return config.DEFAULT_YOLO_MODEL


def main():
    st.title("🚗 TransDetect-Vid — Vehicle Detection (YOLO11)")
    st.write("Tải lên ảnh hoặc video để phát hiện phương tiện giao thông.")

    st.sidebar.header("Cấu hình")
    conf_threshold = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, config.CONF_THRESHOLD, 0.05
    )
    iou_threshold = st.sidebar.slider(
        "IoU threshold", 0.0, 1.0, config.IOU_THRESHOLD, 0.05
    )

    detector = load_detector(_resolve_model_path())

    uploaded = st.file_uploader(
        "Tải lên video hoặc ảnh (.mp4, .avi, .mov, .jpg, .png)",
        type=["mp4", "avi", "mov", "jpg", "jpeg", "png"],
    )
    if uploaded is None:
        return

    ext = uploaded.name.split(".")[-1].lower()
    is_video = ext in ("mp4", "avi", "mov")

    if is_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        tfile.write(uploaded.read())
        tfile.close()

        placeholder = st.empty()
        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect_frame(
                frame, conf_threshold, iou_threshold
            )
            output = visualization.draw_yolo_detections(frame, detections)
            placeholder.image(
                cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
        cap.release()
        try:
            os.remove(tfile.name)
        except OSError:
            pass
        st.success("Đã xử lý xong video.")
    else:
        file_bytes = np.asarray(bytearray(uploaded.getvalue()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        detections = detector.detect_frame(frame, conf_threshold, iou_threshold)
        output = visualization.draw_yolo_detections(frame, detections)
        st.image(
            cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )
        if detections:
            st.subheader("Đối tượng phát hiện được:")
            for det in detections:
                st.write(f"- **{det['class']}** ({det['confidence']:.2%})")
        else:
            st.write("Không phát hiện đối tượng nào trên ngưỡng tin cậy.")


if __name__ == "__main__":
    main()
