import streamlit as st
import json
import base64
import threading
import time
from datetime import datetime

import websocket  # websocket-client library

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
API_WS_URL = "ws://localhost:8000/face/realtime/stream"
API_HTTP_URL = "http://localhost:8000"

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Hệ thống phát hiện khuôn mặt người",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
#  CUSTOM CSS  – Dark Cyberpunk Theme
# ─────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg-dark:    #050810;
    --bg-card:    #0b0f1a;
    --bg-panel:   #0f1626;
    --accent:     #00d4ff;
    --accent2:    #00d4ff;
    --success:    #00ff88;
    --warning:    #ffb800;
    --danger:     #ff3366;
    --text:       #c8d8e8;
    --text-dim:   #4a6080;
    --border:     rgba(0, 212, 255, 0.18);
    --glow:       0 0 20px rgba(0, 212, 255, 0.25);
}

/* ── GLOBAL ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-dark) !important;
    font-family: 'Exo 2', sans-serif;
    color: var(--text);
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 20% 20%, rgba(0,212,255,0.05) 0%, transparent 70%),
        radial-gradient(ellipse 50% 60% at 80% 80%, rgba(123,47,255,0.07) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── MAIN BLOCK ── */
[data-testid="block-container"] {
    padding: 1.5rem 2rem !important;
    max-width: 1400px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.12em !important;

    padding: 0.55rem 1.4rem !important;
    border-radius: 3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.1) !important;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.1) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.35) !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    box-shadow: var(--glow);
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-dim) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.8rem !important;
    color: var(--accent) !important;
}

/* ── SELECTBOX / SLIDER ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-dim) !important;
}

/* ── DIVIDER ── */
hr {
    border-color: var(--border) !important;
    margin: 1.2rem 0 !important;
}

/* ── WS STATUS BADGE ── */
.ws-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
}
.ws-connected    { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid rgba(0,255,136,0.3); }
.ws-disconnected { background: rgba(255,51,102,0.15); color: #ff3366; border: 1px solid rgba(255,51,102,0.3); }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "fps" not in st.session_state:
    st.session_state.fps = 0.0
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False
if "last_frame_b64" not in st.session_state:
    st.session_state.last_frame_b64 = None


# ─────────────────────────────────────────
#  WEBSOCKET CLIENT HELPER
# ─────────────────────────────────────────
def connect_websocket():
    """Tạo kết nối WebSocket đến FastAPI server."""
    try:
        ws = websocket.create_connection(API_WS_URL, timeout=5, enable_multithread=True)
        # Nhận message chào mừng
        welcome = ws.recv()
        return ws
    except Exception as e:
        st.error(f"❌ Không thể kết nối WebSocket: {e}")
        return None


def send_ws_command(ws, command: dict):
    """Gửi lệnh JSON qua WebSocket."""
    try:
        ws.send(json.dumps(command))
    except Exception:
        pass


def receive_ws_message(ws, timeout=0.1):
    """Nhận message từ WebSocket với timeout."""
    try:
        ws.settimeout(timeout)
        raw = ws.recv()
        return json.loads(raw)
    except websocket.WebSocketTimeoutException:
        return None
    except Exception:
        return None


def decode_frame_from_base64(b64_string):
    """Decode base64 string thành bytes cho st.image()."""
    return base64.b64decode(b64_string)


# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown(
    """
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.5rem;">
    <div style="
        width:44px; height:44px; border-radius:8px;
        background:linear-gradient(135deg,#00d4ff22,#00d4ff11);
        border:1px solid rgba(0,212,255,0.4);
        display:flex; align-items:center; justify-content:center;
        font-size:1.4rem; box-shadow:0 0 18px rgba(0,212,255,0.2);">
        🎯
    </div>
    <div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:1.35rem;
                    color:#00d4ff; letter-spacing:0.15em; line-height:1.1;">
            FACE<span style="color:#00d4ff">·</span>ID
        </div>
        <div style="font-size:0.7rem; color:#4a6080; letter-spacing:0.15em;
                    font-family:'Share Tech Mono',monospace;">
            Real-Time Recognition · YOLOv8 · <span class="ws-badge ws-connected">WebSocket API</span>
        </div>
    </div>
</div>
<hr/>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────
#  LAYOUT  – 4 metric pills trên cùng
# ─────────────────────────────────────────
m1, m2 = st.columns(2)
m1.metric("🟢 Trạng thái", "LIVE" if st.session_state.running else "OFFLINE")
m2.metric("⚡ FPS", f"{st.session_state.fps:.1f}")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  CAMERA PANEL
# ─────────────────────────────────────────

st.markdown(
    """
<div style="
    background:var(--bg-panel);
    border:1px solid var(--border);
    border-radius:8px;
    padding:1rem 1.2rem 0.8rem;
    margin-bottom:0.8rem;
    box-shadow:var(--glow);">
    <span style="font-family:'Share Tech Mono',monospace; font-size:0.75rem;
                 color:#4a6080; letter-spacing:0.15em;">
        ▸ camera feed — <span class="ws-badge ws-connected" id="ws-status">VIA WEBSOCKET API</span>
    </span>
</div>
""",
    unsafe_allow_html=True,
)

frame_placeholder = st.empty()

# Placeholder khi chưa chạy
if not st.session_state.running:
    frame_placeholder.markdown(
        """
    <div style="
        background:var(--bg-panel);
        border:1px dashed rgba(0,212,255,0.2);
        border-radius:6px;
        height:380px;
        display:flex; flex-direction:column;
        align-items:center; justify-content:center;
        gap:0.6rem;">
        <div style="font-size:3rem; opacity:0.25;">📷</div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem;
                    color:#4a6080; letter-spacing:0.15em;">
            CAMERA OFFLINE
        </div>
        <div style="font-size:0.72rem; color:#2a3a50;">
            Nhấn START để kết nối WebSocket API và bắt đầu nhận diện
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Controls
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
cb1, cb2 = st.columns(2)
with cb1:
    start_btn = st.button("▶  START", use_container_width=True)
with cb2:
    stop_btn = st.button("■  STOP", use_container_width=True)

if start_btn:
    st.session_state.running = True
    st.rerun()
if stop_btn:
    st.session_state.running = False
    st.rerun()

# ── SETTINGS (collapsible) ──
with st.expander("⚙  Cài đặt phát hiện"):
    conf_thresh = st.slider("Ngưỡng tin cậy (confidence)", 0.1, 1.0, 0.5, 0.05)
    cam_index = st.selectbox(
        "Nguồn camera", [0, 1], format_func=lambda x: f"Webcam {x}"
    )


# ─────────────────────────────────────────
#  LIVE CAMERA LOOP — VIA WEBSOCKET API
# ─────────────────────────────────────────
if st.session_state.running:
    # ── Kết nối WebSocket đến FastAPI server ──
    ws = connect_websocket()

    if ws is None:
        st.error(
            "❌ Không thể kết nối đến API server. Hãy chắc chắn FastAPI đang chạy:\n\n`uvicorn main:app --reload`"
        )
        st.session_state.running = False
    else:
        st.session_state.ws_connected = True

        try:
            # Gửi lệnh START stream qua WebSocket
            camera_source = cam_index if "cam_index" in dir() else 0
            send_ws_command(ws, {"action": "start", "camera_index": camera_source})

            # Nhận phản hồi start
            start_response = receive_ws_message(ws, timeout=5)
            if start_response and start_response.get("status") == "error":
                st.error(f"❌ {start_response.get('message', 'Camera error')}")
                ws.close()
                st.session_state.running = False
            else:
                # Gửi config nếu cần
                if "conf_thresh" in dir() and conf_thresh != 0.5:
                    send_ws_command(
                        ws, {"action": "config", "conf_threshold": conf_thresh}
                    )
                    receive_ws_message(ws, timeout=1)  # consume response

                # ── MAIN LOOP: nhận frames từ WebSocket ──
                while st.session_state.running:
                    msg = receive_ws_message(ws, timeout=2)

                    if msg is None:
                        continue

                    msg_type = msg.get("type", "")

                    if msg_type == "frame":
                        # Decode frame base64 → hiển thị
                        frame_bytes = decode_frame_from_base64(msg["frame"])
                        frame_placeholder.image(frame_bytes, use_container_width=True)

                        # Cập nhật FPS
                        st.session_state.fps = msg.get("fps", 0.0)

                    elif msg_type == "status":
                        if msg.get("status") == "error":
                            st.warning(f"⚠ API: {msg.get('message', '')}")
                            break

        except Exception as e:
            st.error(f"❌ WebSocket error: {e}")
        finally:
            # ── Cleanup: gửi stop và đóng WebSocket ──
            try:
                send_ws_command(ws, {"action": "stop"})
                time.sleep(0.1)
                ws.close()
            except Exception:
                pass
            st.session_state.ws_connected = False
