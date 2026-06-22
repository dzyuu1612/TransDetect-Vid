
import streamlit as st
import cv2
import tempfile
import os
import yolo_detector

st.set_page_config(page_title="Vehicle Detection UI", page_icon="🏍️")

@st.cache_resource
def load_yolo_model():
    """Load model once and cache it for speed"""
    # Using the best.pt from our training run
    model_path = r"runs\detect\motorbike_yolo11n\weights\best.pt"
    if not os.path.exists(model_path):
        fallback_path = "yolo11n.pt"
        if os.path.exists(fallback_path):
            st.warning(f"Chưa tìm thấy model đã train ở `{model_path}`. Đang dùng tạm model mặc định `{fallback_path}`.")
            return yolo_detector.load_model(fallback_path)
        else:
            st.error(f"Model not found at {model_path}. Please wait for training to generate it.")
            return None
    return yolo_detector.load_model(model_path)

def main():
    st.title("🏍️ Motorbike & Car Detector (YOLO11)")
    st.write("Upload a video to see real-time object detection!")

    # Sidebar settings
    st.sidebar.header("Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    custom_fps = st.sidebar.slider("Playback FPS (0 = original video speed)", min_value=0.0, max_value=120.0, value=0.0, step=5.0)
    frame_skip = st.sidebar.slider("Frame Skip (Process 1 in every N frames)", min_value=1, max_value=10, value=1, step=1)
    
    # Load model
    model = load_yolo_model()
    
    if model is None:
        return

    # File uploader
    uploaded_file = st.file_uploader("Upload a Video or Image (.mp4, .avi, .mov, .jpg, .png)", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        is_video = file_extension in ['mp4', 'avi', 'mov']

        if is_video:
            # Save the uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
            tfile.write(uploaded_file.read())
            tfile.close()

            st.success("Video uploaded successfully! Processing...")

            # Create a placeholder for the video frame
            frame_placeholder = st.empty()
            
            # Open the video with OpenCV
            cap = cv2.VideoCapture(tfile.name)
            
            # Get the original video FPS
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if not original_fps or original_fps == 0:
                original_fps = 25.0
            
            target_fps = custom_fps if custom_fps > 0 else original_fps
            frame_time = 1.0 / target_fps
            
            import time
            stop_button = st.button("Stop Processing")
            frame_counter = 0

            while cap.isOpened():
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret or stop_button:
                    break
                    
                frame_counter += 1
                if frame_counter % frame_skip != 0:
                    continue
                    
                # OpenCV reads in BGR, but Streamlit expects RGB
                # Let's detect on the BGR frame first
                boxes = yolo_detector.detect_frame(model, frame, conf=conf_threshold)
                output_frame = yolo_detector.draw_boxes(frame, boxes)
                
                # Convert BGR to RGB for Streamlit displaying
                output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                
                # Show the frame
                frame_placeholder.image(output_frame_rgb, channels="RGB", use_container_width=True)
                
                # Sleep if we processed the frame faster than the original video FPS
                elapsed = time.time() - start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                
            cap.release()
            
            # Cleanup temporary file
            try:
                os.remove(tfile.name)
            except:
                pass
                
            if not stop_button:
                st.success("Video processing completed!")
        else:
            # It's an image
            import numpy as np
            
            # Read image directly from uploaded file using getvalue() to avoid empty reads
            file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            st.success("Image uploaded successfully! Detecting...")
            
            # Detect on the image
            boxes = yolo_detector.detect_frame(model, frame, conf=conf_threshold)
            output_frame = yolo_detector.draw_boxes(frame, boxes)
            
            # Convert BGR to RGB for Streamlit
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            
            # Display the result (using use_container_width as it's still widely supported despite the future warning)
            st.image(output_frame_rgb, channels="RGB", use_container_width=True)
            st.success("Image detection completed!")

            # Add a text summary below the image
            if len(boxes) > 0:
                st.subheader("Objects Detected:")
                for i, box in enumerate(boxes):
                    # boxes tuple: (x1, y1, x2, y2, class_name, confidence)
                    class_name = box[4]
                    confidence = box[5]
                    st.write(f"- **{class_name}** (Confidence: {confidence:.2%})")
            else:
                st.write("No objects detected above the confidence threshold.")

if __name__ == "__main__":
    main()
