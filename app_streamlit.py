"""
app_streamlit.py
------------------
TransDetect-Vid Dashboard — Mock UI with Real-time Simulation.

File này chứa giao diện với mock data, kết hợp vòng lặp mô phỏng
đọc frame từ video thật thông qua OpenCV để kiểm tra layout thời gian thực.
Chạy:  streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import cv2
import time
import tempfile
import os
import sys

# Add project root to path so we can import src package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.transdetect import config, preprocessing, classical_detector, visualization
from src.transdetect.yolo_detector import Yolo11VehicleDetector

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TransDetect-Vid",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "is_paused" not in st.session_state:
    st.session_state.is_paused = False
if "current_frame_idx" not in st.session_state:
    st.session_state.current_frame_idx = 0
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "results_history" not in st.session_state:
    st.session_state.results_history = []
if "export_data_list" not in st.session_state:
    st.session_state.export_data_list = []

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
CUSTOM_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
.block-container { padding-top: 0.6rem; padding-bottom: 1rem; }
[data-testid="stAppViewContainer"] > div:first-child { background: #f0f2f6; }

/* Header Bar */
.header-bar {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 14px 28px; display: flex; align-items: center; justify-content: space-between;
    border-radius: 12px; margin-bottom: 18px; box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}
.header-left { display: flex; align-items: center; gap: 14px; }
.header-icon {
    width: 40px; height: 40px; background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-size: 20px; color: white;
}
.header-title { color: #f8fafc; font-size: 20px; font-weight: 700; margin: 0; line-height: 1.2; }
.header-subtitle { color: #94a3b8; font-size: 12px; margin: 0; }
.header-right { display: flex; align-items: center; gap: 10px; }
.badge-ready {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3);
    color: #10b981; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;
}
.badge-dot { width: 8px; height: 8px; background: #10b981; border-radius: 50%; display: inline-block; }
.hdr-btn-secondary {
    padding: 8px 18px; border-radius: 8px; font-size: 13px; font-weight: 600; border: none; cursor: pointer;
    background: rgba(255,255,255,0.08); color: #e2e8f0; border: 1px solid rgba(255,255,255,0.1);
}

/* Sections & Cards */
.section-title { font-size: 14px; font-weight: 700; color: #1e293b; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #e2e8f0; }
.vehicle-card { border-radius: 10px; padding: 12px 8px; text-align: center; margin-bottom: 8px; }
.vehicle-card-blue { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #93c5fd; }
.vehicle-card-green { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border: 1px solid #6ee7b7; }
.vehicle-icon { font-size: 22px; }
.vehicle-label { font-size: 11px; font-weight: 600; color: #64748b; margin: 4px 0 2px 0; }
.vehicle-count { font-size: 26px; font-weight: 800; line-height: 1.1; margin: 0; }
.vehicle-count-blue { color: #1d4ed8; }
.vehicle-count-green { color: #15803d; }

/* FPS */
.fps-container { text-align: center; padding: 8px 0; }
.fps-label { font-size: 11px; color: #94a3b8; font-weight: 600; text-transform: uppercase; margin: 0 0 2px 0; }
.fps-value { font-size: 48px; font-weight: 800; color: #2563eb; line-height: 1; margin: 0; }
.fps-unit { font-size: 18px; font-weight: 400; color: #94a3b8; margin-left: 4px; }
.info-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 13px; }
.info-label { color: #64748b; font-weight: 500; }
.info-value { color: #1e293b; font-weight: 600; }

/* Video */
.video-preview { background: #0f172a; border-radius: 10px 10px 0 0; min-height: 340px; display: flex; align-items: center; justify-content: center; }
.video-controls { background: #0f172a; padding: 10px 16px; border-radius: 0 0 10px 10px; display: flex; align-items: center; gap: 12px; }
.vc-time { color: #e2e8f0; font-size: 12px; font-family: monospace; }
.vc-progress-bar { flex: 1; height: 5px; background: #334155; border-radius: 3px; position: relative; overflow: hidden; }
.vc-progress-fill { position: absolute; left: 0; top: 0; height: 100%; background: linear-gradient(90deg, #3b82f6, #06b6d4); border-radius: 3px; }
.vc-speed { color: #e2e8f0; font-size: 12px; background: rgba(255,255,255,0.08); padding: 2px 10px; border-radius: 4px; }

/* Table */
.results-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 4px; background: white; border-radius: 8px; overflow: hidden; }
.results-table th { background: #f8fafc; color: #64748b; font-weight: 600; padding: 8px; text-align: center; white-space: nowrap; }
.results-table th.group-header { font-size: 13px; color: #1e293b; font-weight: 700; border-bottom: 1px solid #e2e8f0; }
.results-table th.sub-header { font-size: 11px; border-bottom: 2px solid #e2e8f0; }
.results-table td { padding: 7px 8px; text-align: center; border-bottom: 1px solid #f1f5f9; color: #334155; font-size: 12px; }
.results-table tr:hover { background: #f8fafc; }
.results-table .row-highlight { background: #eff6ff; }
.results-table .row-highlight td { color: #1e40af; font-weight: 600; }
[data-testid="stColumn"] > div > div > div > div { background-color: transparent; }
</style>"""

# ═══════════════════════════════════════════════════════════════
# HTML RENDER HELPERS
# ═══════════════════════════════════════════════════════════════
def _build_results_table(rows, highlight_row_idx=0):
    if not rows:
        return '<div style="padding: 20px; text-align: center; color: #94a3b8; font-size: 13px; border: 1px dashed #cbd5e1; border-radius: 8px;">No data yet. Run detection.</div>'
        
    html = '<table class="results-table"><thead><tr>'
    html += '<th rowspan="2" class="group-header" style="border-bottom:2px solid #e2e8f0">Frame</th>'
    html += '<th rowspan="2" class="group-header" style="border-bottom:2px solid #e2e8f0">Time (s)</th>'
    for name in ("Car", "Motorcycle", "Bus", "Truck"):
        html += f'<th colspan="2" class="group-header">{name}</th>'
    html += '<th rowspan="2" class="group-header" style="border-bottom:2px solid #e2e8f0">Total</th></tr><tr>'
    for _ in range(4):
        html += '<th class="sub-header">Count</th><th class="sub-header">Avg Conf</th>'
    html += "</tr></thead><tbody>"
    
    for i, r in enumerate(rows):
        cls = ' class="row-highlight"' if i == highlight_row_idx else ""
        html += f"<tr{cls}><td>{r[0]}</td><td>{r[1]}</td>"
        for j in range(2, 10, 2):
            html += f"<td>{r[j]}</td><td>{r[j+1]:.2f}</td>"
        html += f"<td>{r[10]}</td></tr>"
    html += "</tbody></table>"
    return html

def render_vehicle_card(type_str, count, color_theme):
    icon = {"Car": "🚗", "Motorcycle": "🏍️", "Bus": "🚌", "Truck": "🚛"}.get(type_str, "")
    return f"""
    <div class="vehicle-card vehicle-card-{color_theme}">
        <div class="vehicle-icon">{icon}</div>
        <p class="vehicle-label">{type_str}</p>
        <p class="vehicle-count vehicle-count-{color_theme}">{count}</p>
    </div>
    """

def render_video_info(name, duration_str, fps, total_frames, processed_frames):
    pct = (processed_frames / total_frames * 100) if total_frames > 0 else 0
    return f"""
        <div class="info-row"><span class="info-label">Video Name</span><span class="info-value">{name}</span></div>
        <div class="info-row"><span class="info-label">Duration</span><span class="info-value">{duration_str}</span></div>
        <div class="info-row"><span class="info-label">Frame Rate</span><span class="info-value">{fps:.2f} FPS</span></div>
        <div class="info-row"><span class="info-label">Total Frames</span><span class="info-value">{total_frames}</span></div>
        <div class="info-row"><span class="info-label">Processed</span><span class="info-value">{processed_frames} / {total_frames} ({pct:.1f}%)</span></div>
    """

def render_header():
    st.markdown("""
    <div class="header-bar">
        <div class="header-left">
            <div class="header-icon">⚙</div>
            <div>
                <p class="header-title">TransDetect-Vid</p>
                <p class="header-subtitle">Vehicle Detection in Video</p>
            </div>
        </div>
        <div class="header-right">
            <span class="badge-ready"><span class="badge-dot"></span> System Ready</span>
            <span class="hdr-btn hdr-btn-secondary">ⓘ About</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_header()

    col1, col2, col3 = st.columns([1, 2, 1], gap="medium")

    # ─── COLUMN 1: CONTROL PANEL ───
    with col1:
        with st.container(border=True):
            st.markdown('<p class="section-title">Video Source</p>', unsafe_allow_html=True)
            
            # 1. Local files in test_videos
            test_videos_dir = os.path.join(os.path.dirname(__file__), "test_videos")
            if not os.path.exists(test_videos_dir):
                try: os.makedirs(test_videos_dir, exist_ok=True)
                except: pass
                
            local_videos = ["-- Select a local video --"]
            if os.path.exists(test_videos_dir):
                local_videos += [f for f in os.listdir(test_videos_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
                
            selected_local = st.selectbox("Select from 'test_videos' folder", local_videos)
            
            st.markdown("<div style='text-align:center; font-size:12px; margin: -5px 0 5px 0; color: #64748b;'>OR</div>", unsafe_allow_html=True)
            
            # 2. File Uploader
            uploaded_file = st.file_uploader("Upload new video", type=["mp4", "avi", "mov"])
            
            # Logic to handle both
            if uploaded_file is not None:
                if st.session_state.uploaded_file_name != uploaded_file.name:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_file.read())
                    st.session_state.uploaded_video_path = tfile.name
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.results_history = []
                    st.session_state.export_data_list = []
                    st.session_state.is_running = False
            elif selected_local != "-- Select a local video --":
                if st.session_state.uploaded_file_name != selected_local:
                    st.session_state.uploaded_video_path = os.path.join(test_videos_dir, selected_local)
                    st.session_state.uploaded_file_name = selected_local
                    st.session_state.results_history = []
                    st.session_state.export_data_list = []
                    st.session_state.is_running = False

            st.divider()
            st.markdown('<p class="section-title">Detection Method</p>', unsafe_allow_html=True)
            st.radio("method", ["Classical Pipeline", "YOLO11"], captions=["Threshold + Sobel + Contour", "Ultralytics YOLO11"], label_visibility="collapsed", key="method_selector")

            st.divider()
            st.markdown('<p class="section-title">Parameters</p>', unsafe_allow_html=True)
            st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01, key="conf_thresh")
            st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.01, key="iou_thresh")
            st.number_input("Max Detection per Frame", 1, 500, 100, 10, key="max_det")

            st.divider()
            st.markdown('<p class="section-title">Target Classes</p>', unsafe_allow_html=True)
            st.checkbox("🚗  Car (2)", value=True)
            st.checkbox("🏍️  Motorcycle (3)", value=True)
            st.checkbox("🚌  Bus (5)", value=True)
            st.checkbox("🚛  Truck (7)", value=True)

            st.divider()
            c_run, c_pause, c_stop = st.columns(3)
            with c_run:
                if st.button("▶ Run", type="primary", use_container_width=True):
                    if st.session_state.uploaded_video_path:
                        st.session_state.is_running = True
                        st.session_state.is_paused = False
                        st.session_state.current_frame_idx = 0
                        st.session_state.results_history = []
                        st.session_state.export_data_list = []
                    else:
                        st.error("Upload a video first!")
            with c_pause:
                pause_label = "▶ Resume" if st.session_state.get("is_paused") else "⏸ Pause"
                if st.button(pause_label, use_container_width=True):
                    if st.session_state.is_running:
                        st.session_state.is_paused = not st.session_state.get("is_paused", False)
            with c_stop:
                if st.button("⬜ Stop", use_container_width=True):
                    st.session_state.is_running = False
                    st.session_state.is_paused = False
                    st.session_state.current_frame_idx = 0

    # ─── COLUMN 2: VIDEO PREVIEW ───
    with col2:
        with st.container(border=True):
            st.markdown('<p class="section-title">Video Preview</p>', unsafe_allow_html=True)
            
            # Placeholders
            time_text_placeholder = st.empty()
            progress_bar_placeholder = st.empty()
            video_placeholder = st.empty()
            controls_placeholder = st.empty()

            # Default empty state
            if not st.session_state.is_running:
                if not st.session_state.uploaded_video_path:
                    video_placeholder.markdown('<div class="video-preview"><div style="text-align:center;"><div style="font-size:48px; margin-bottom:12px;">🎬</div><div style="color:#64748b; font-size:15px;">Upload a video to start</div></div></div>', unsafe_allow_html=True)
                else:
                    video_placeholder.markdown('<div class="video-preview"><div style="text-align:center;"><div style="font-size:48px; margin-bottom:12px;">▶️</div><div style="color:#64748b; font-size:15px;">Ready to run detection</div></div></div>', unsafe_allow_html=True)
                controls_placeholder.markdown('<div class="video-controls"><span style="color:#fff; font-size:14px;">▶</span><span class="vc-time">00:00:00 / 00:00:00</span><div class="vc-progress-bar"><div class="vc-progress-fill" style="width:0%"></div></div><span class="vc-speed">1.0x</span></div>', unsafe_allow_html=True)

            st.write("")
            hc, cc, jc = st.columns([3, 1, 1])
            
            # Generate CSV data
            if st.session_state.get("export_data_list"):
                df = pd.DataFrame(st.session_state.export_data_list)
                csv_data = df.to_csv(index=False)
            else:
                csv_data = "frame_id,class_name,confidence,count\n"

            with hc: st.markdown("**Detection Results** (Preview)")
            with cc: st.download_button("⬇ CSV", csv_data, "res.csv", use_container_width=True)
            with jc: st.download_button("{ } JSON", "{}", "res.json", use_container_width=True)

            table_placeholder = st.empty()
            if not st.session_state.is_running:
                table_placeholder.markdown(_build_results_table(st.session_state.results_history), unsafe_allow_html=True)

    # ─── COLUMN 3: ANALYTICS ───
    with col3:
        with st.container(border=True):
            st.markdown('<p class="section-title">Performance</p>', unsafe_allow_html=True)
            fps_placeholder = st.empty()
            perf_c1, perf_c2 = st.columns(2)
            with perf_c1: avg_fps_metric = st.empty()
            with perf_c2: res_metric = st.empty()

            st.divider()
            st.markdown('<p class="section-title">Detected Vehicles <span style="color:#94a3b8;font-size:11px;">(Current)</span></p>', unsafe_allow_html=True)
            vc1, vc2 = st.columns(2)
            with vc1: car_placeholder = st.empty()
            with vc2: moto_placeholder = st.empty()
            vc3, vc4 = st.columns(2)
            with vc3: bus_placeholder = st.empty()
            with vc4: truck_placeholder = st.empty()

            st.divider()
            st.markdown('<p class="section-title">Video Information</p>', unsafe_allow_html=True)
            info_placeholder = st.empty()
            progress_bar = st.empty()

            # Default static values
            if not st.session_state.is_running:
                fps_placeholder.markdown('<div class="fps-container"><p class="fps-label">FPS (Current)</p><p><span class="fps-value">0.0</span><span class="fps-unit">fps</span></p></div>', unsafe_allow_html=True)
                avg_fps_metric.metric("Average FPS", "0.0 fps")
                res_metric.metric("Resolution", "-")
                
                car_placeholder.markdown(render_vehicle_card("Car", 0, "blue"), unsafe_allow_html=True)
                moto_placeholder.markdown(render_vehicle_card("Motorcycle", 0, "blue"), unsafe_allow_html=True)
                bus_placeholder.markdown(render_vehicle_card("Bus", 0, "green"), unsafe_allow_html=True)
                truck_placeholder.markdown(render_vehicle_card("Truck", 0, "green"), unsafe_allow_html=True)
                
                info_placeholder.markdown(render_video_info(st.session_state.uploaded_file_name or "-", "00:00:00", 0, 0, 0), unsafe_allow_html=True)
                progress_bar.progress(0.0)

    # ═══════════════════════════════════════════════════════════════
    # REAL-TIME SIMULATION LOOP
    # ═══════════════════════════════════════════════════════════════
    if st.session_state.is_running and st.session_state.uploaded_video_path:
        cap = cv2.VideoCapture(st.session_state.uploaded_video_path)
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        duration_s = total_frames / fps_video if fps_video > 0 else 0
        dur_str = time.strftime('%H:%M:%S', time.gmtime(duration_s))

        res_metric.metric("Resolution", f"{w} × {h}")

        if st.session_state.get("is_paused", False):
            if st.session_state.current_frame_idx > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_idx)
            ret, frame = cap.read()
            if ret:
                video_placeholder.image(frame, channels="BGR")
            cap.release()
            return # Exit to keep Streamlit responsive while paused

        if st.session_state.get("current_frame_idx", 0) > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_idx)

        frame_idx = st.session_state.get("current_frame_idx", 0)
        start_frame_idx = frame_idx
        start_time = time.time()
        last_ui_update_time = 0
        last_fps_time = time.time()
        last_fps_frame_idx = frame_idx
        instant_fps = 0.0
        
        yolo_detector = None
        if st.session_state.method_selector == "YOLO11":
            try:
                # Load YOLO model only once before loop
                yolo_detector = Yolo11VehicleDetector(config.DEFAULT_YOLO_MODEL)
            except Exception as e:
                st.error(f"Failed to load YOLO model: {e}")
                st.session_state.is_running = False

        while cap.isOpened() and st.session_state.is_running and not st.session_state.get("is_paused", False):
            ret, frame = cap.read()
            if not ret:
                st.session_state.is_running = False
                st.session_state.current_frame_idx = 0
                break
                
            frame_idx += 1
            st.session_state.current_frame_idx = frame_idx
            curr_time = time.time()
            elapsed = curr_time - start_time
            
            # Calculate instantaneous FPS over the last 10 frames to avoid warmup penalty
            if frame_idx - last_fps_frame_idx >= 10:
                instant_fps = (frame_idx - last_fps_frame_idx) / (curr_time - last_fps_time)
                last_fps_time = curr_time
                last_fps_frame_idx = frame_idx
            
            current_fps = instant_fps if instant_fps > 0 else 0
            
            # --- 1. Detection Logic (REAL DATA) ---
            detections = []
            if st.session_state.method_selector == "Classical Pipeline":
                pre = preprocessing.preprocess_frame(frame, config.MEDIAN_KERNEL_SIZE)
                boxes, _, _ = classical_detector.detect_vehicle_candidates(
                    pre, min_area=config.MIN_CONTOUR_AREA, max_aspect_ratio=config.MAX_ASPECT_RATIO
                )
                counts = {"Car": len(boxes), "Motorcycle": 0, "Bus": 0, "Truck": 0}
                frame_to_draw = visualization.draw_classical_boxes(frame.copy(), boxes)
            else:
                if yolo_detector:
                    detections = yolo_detector.detect_frame(frame, st.session_state.conf_thresh, st.session_state.iou_thresh)
                    counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
                    for det in detections:
                        cls_name = str(det.get("class", "Car")).title()
                        if cls_name in counts:
                            counts[cls_name] += 1
                        else:
                            counts["Car"] += 1
                            
                    frame_to_draw = visualization.draw_yolo_detections(frame.copy(), detections)
                else:
                    frame_to_draw = frame.copy()
                    counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

            c_car = counts.get("Car", 0)
            c_moto = counts.get("Motorcycle", 0)
            c_bus = counts.get("Bus", 0)
            c_truck = counts.get("Truck", 0)
            c_tot = c_car + c_moto + c_bus + c_truck

            # --- Export Data Collection ---
            for cls_name, count in counts.items():
                if count > 0:
                    avg_conf = 1.0
                    if st.session_state.method_selector == "YOLO11" and yolo_detector:
                        confs = [d["confidence"] for d in detections if str(d.get("class", "Car")).title() == cls_name]
                        if confs:
                            avg_conf = sum(confs) / len(confs)
                    
                    st.session_state.export_data_list.append({
                        "frame_id": frame_idx,
                        "class_name": cls_name,
                        "confidence": round(avg_conf, 2),
                        "count": count
                    })

            # --- 2. Update UI Preview (Throttled to prevent browser freeze) ---
            if curr_time - last_ui_update_time > 0.03: # Max ~33 FPS UI refresh
                curr_s = frame_idx / fps_video if fps_video > 0 else 0
                curr_str = time.strftime('%H:%M:%S', time.gmtime(curr_s))
                prog_val = min(1.0, frame_idx / total_frames) if total_frames > 0 else 0.0
                
                time_text_placeholder.markdown(f"▶️ **{curr_str} / {dur_str}**")
                progress_bar_placeholder.progress(prog_val)
                
                # Resize image for UI to reduce WebSocket payload and prevent freeze
                h_ui, w_ui = frame_to_draw.shape[:2]
                if h_ui > 480:
                    scale = 480 / h_ui
                    frame_to_draw_ui = cv2.resize(frame_to_draw, (int(w_ui * scale), 480))
                else:
                    frame_to_draw_ui = frame_to_draw
                    
                video_placeholder.image(frame_to_draw_ui, channels="BGR")
                
                # --- 3. Update Controls ---
                controls_placeholder.markdown(f'<div class="video-controls"><span style="color:#fff; font-size:14px;">▶</span><span class="vc-time">{curr_str} / {dur_str}</span><div class="vc-progress-bar"><div class="vc-progress-fill" style="width:{prog_val*100}%"></div></div><span class="vc-speed">1.0x</span></div>', unsafe_allow_html=True)
                
                # --- 4. Update Metrics ---
                fps_placeholder.markdown(f'<div class="fps-container"><p class="fps-label">FPS (Current)</p><p><span class="fps-value">{current_fps:.1f}</span><span class="fps-unit">fps</span></p></div>', unsafe_allow_html=True)
                avg_fps_metric.metric("Average FPS", f"{current_fps:.1f} fps")
                
                car_placeholder.markdown(render_vehicle_card("Car", c_car, "blue"), unsafe_allow_html=True)
                moto_placeholder.markdown(render_vehicle_card("Motorcycle", c_moto, "blue"), unsafe_allow_html=True)
                bus_placeholder.markdown(render_vehicle_card("Bus", c_bus, "green"), unsafe_allow_html=True)
                truck_placeholder.markdown(render_vehicle_card("Truck", c_truck, "green"), unsafe_allow_html=True)
                
                # --- 4. Update Table ---
                hist = (frame_idx, f"{curr_s:.2f}", c_car, 0.85, c_moto, 0.9, c_bus, 0.88, c_truck, 0.9, c_tot)
                st.session_state.results_history.insert(0, hist)
                if len(st.session_state.results_history) > 6:
                    st.session_state.results_history.pop()
                table_placeholder.markdown(_build_results_table(st.session_state.results_history, highlight_row_idx=0), unsafe_allow_html=True)

                # --- 5. Update Info & Progress ---
                info_placeholder.markdown(render_video_info(st.session_state.uploaded_file_name, dur_str, fps_video, total_frames, frame_idx), unsafe_allow_html=True)
                progress_bar.progress(prog_val)
                
                last_ui_update_time = curr_time

            # --- 6. Real-time Throttling ---
            # Wait to match the original video FPS
            if fps_video > 0:
                expected_elapsed = (frame_idx - start_frame_idx) / fps_video
                actual_elapsed = time.time() - start_time
                if actual_elapsed < expected_elapsed:
                    time.sleep(expected_elapsed - actual_elapsed)

        cap.release()


if __name__ == "__main__":
    main()
