# YÊU CẦU XÂY DỰNG GIAO DIỆN STREAMLIT CHO TRANSDETECT-VID

## Mục tiêu
Build một dashboard Streamlit dựa trên layout 3 cột, bám sát cấu trúc của hình ảnh giao diện mẫu được cung cấp. Ưu tiên sự gọn gàng, chia khu vực rõ ràng và dễ nhìn.

## Cấu trúc Layout cơ bản
Sử dụng `st.set_page_config(layout="wide")` làm mặc định để xài full màn hình.
Chia màn hình thành 3 phần bằng `col1, col2, col3 = st.columns([1, 2, 1])` (Cột giữa to gấp đôi 2 cột bìa).

### Cột 1: Control Panel (Cột trái)
- Phân tách các phần bằng `st.divider()` hoặc `st.container()`.
- **Detection Method:** Dùng `st.radio` (Classical Pipeline, YOLO11).
- **Parameters:** - Dùng `st.slider` cho Confidence Threshold (0.00-1.00) và IoU Threshold (0.00-1.00).
  - Dùng `st.number_input` cho Max Detection per Frame.
- **Target Classes:** Dùng `st.checkbox` cho Car, Motorcycle, Bus, Truck.
- **Action Buttons:** Dùng `st.button` ("Run Detection" và "Stop").

### Cột 2: Main Stage (Cột giữa - Khu vực to nhất)
- **Video Preview:** Dùng `st.empty()` để tạo placeholder. Nơi này sẽ liên tục update frame bằng `st.image` khi render real-time bằng OpenCV.
- **Detection Results (Preview):** Dùng `st.dataframe` để show bảng thống kê dữ liệu.
- Cần layout ngang (st.columns nhỏ) để nhét 2 nút `st.download_button` (Download CSV, Download JSON) nằm ngay trên cái dataframe.

### Cột 3: Analytics & System Info (Cột phải)
- Sử dụng chủ yếu `st.metric` để tạo các thông số.
- **Performance:** `st.metric` show FPS và Average FPS.
- **Detected Vehicles (Current Frame):** Dùng `st.columns(2)` bên trong cột 3 để tạo lưới 2x2. Show 4 cái `st.metric` cho số lượng xe (Car, Motorcycle, Bus, Truck).
- **Video Information:** Hiển thị thông số tĩnh (Video Name, Duration, v.v.) và dùng `st.progress` cho thanh tiến trình xử lý.

## Yêu cầu kỹ thuật cho Agent
- KHÔNG viết code dính chặt với logic core của hệ thống vội. Hãy tạo file `app_streamlit.py` thuần UI mock data trước để user review layout.
- Tận dụng tối đa component native của Streamlit. Chỉ dùng Custom CSS (`st.markdown`) cho những chỗ thực sự cần thiết (như đổi màu text, chỉnh padding).
- Viết code theo dạng module, chia thành các hàm tạo UI riêng biệt (ví dụ: `render_sidebar()`, `render_main_video_area()`, `render_analytics()`) để file không bị rác.