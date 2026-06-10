# Giao diện web bằng Streamlit

import streamlit as st
import cv2
import time
import numpy as np
import os

from src.domain.face_detector import FaceDetector
from src.utils.video_utils import VideoStream, calculate_fps

st.set_page_config(page_title="Face Detection with YOLOv8", layout="wide")
st.title("Face Detection with YOLOv8")

# Tạo thư mục images nếu chưa có
os.makedirs("images", exist_ok=True)

# Khởi tạo session state
if "video_started" not in st.session_state:
    st.session_state.video_started = False

if "stream" not in st.session_state:
    st.session_state.stream = None


@st.cache_resource
def load_detector():
    return FaceDetector()


detector = load_detector()

# Sidebar
st.sidebar.header("Video Source")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

col_start, col_stop = st.sidebar.columns(2)

with col_start:
    start_button = st.button("Start Video", use_container_width=True)

with col_stop:
    stop_button = st.button("Stop Video", use_container_width=True)

if start_button and not st.session_state.video_started:
    st.session_state.video_started = True

if stop_button and st.session_state.video_started:
    st.session_state.video_started = False


# =========================
# REALTIME CAMERA DETECTION
# =========================
if st.session_state.video_started:
    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    # Dừng stream cũ nếu còn tồn tại
    if st.session_state.stream is not None:
        st.session_state.stream.stop()
        st.session_state.stream = None

    stream = VideoStream(source=0)
    stream.start()
    st.session_state.stream = stream

    prev_time = time.time()

    while st.session_state.video_started:
        ret, frame = stream.read()

        if not ret or frame is None:
            time.sleep(0.01)
            continue

        annotated, detections = detector.detect(frame)
        fps, prev_time = calculate_fps(prev_time)

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

        info_placeholder.markdown(
            f"**FPS:** {fps:.2f} | "
            f"**Faces:** {len(detections)} | "
            f"**Confidence:** {conf_threshold:.2f}"
        )

    # Dừng stream khi thoát loop
    if st.session_state.stream is not None:
        st.session_state.stream.stop()
        st.session_state.stream = None

else:
    st.info("Click 'Start Video' to begin face detection.")


# =========================
# UPLOAD IMAGE DETECTION
# =========================
st.sidebar.markdown("---")
st.sidebar.header("Upload image")

uploaded_file = st.sidebar.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Lưu file gốc
    save_path = os.path.join("images", uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully.")

    # Đọc ảnh thành numpy array
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        # Detect khuôn mặt bằng YOLOv8
        annotated, detections = detector.detect(image)

        # Hiển thị kết quả
        st.subheader(f"🔍 Detected {len(detections)} face(s)")

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(
            annotated_rgb,
            caption=f"Detection result - {uploaded_file.name}",
            use_container_width=True
        )

        # Hiển thị chi tiết từng khuôn mặt
        if detections:
            for i, det in enumerate(detections):
                bbox = det["bbox"]
                conf = det["confidence"]

                st.markdown(
                    f"**Face {i + 1}:** Confidence `{conf:.2%}` | "
                    f"Position `({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})`"
                )
        else:
            st.warning("Không phát hiện khuôn mặt nào trong ảnh.")
    else:
        st.error("Không thể đọc file ảnh.")