# WebSocket API endpoint cho realtime face detection
import asyncio
import base64
import json
import time
import cv2
import numpy as np
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.domain.face_detector import FaceDetector
from src.utils.video_utils import VideoStream, calculate_fps

router = APIRouter(prefix="/face/realtime", tags=["Realtime Face Detection"])
detector = FaceDetector()


@router.websocket("/stream")
async def realtime_face_stream(websocket: WebSocket):
    """
    WebSocket endpoint cho realtime face detection.

    Protocol (bidirectional):
    ─────────────────────────────────────────────────────
    CLIENT → SERVER (JSON commands):
        {"action": "start"}                         — Bắt đầu stream camera
        {"action": "start", "camera_index": 1}      — Chọn camera
        {"action": "stop"}                          — Dừng stream
        {"action": "snapshot"}                      — Chụp 1 frame
        {"action": "config", "conf_threshold": 0.6} — Thay đổi cấu hình

    SERVER → CLIENT (JSON responses):
        Frame data:
        {
            "type": "frame",
            "frame": "<base64 JPEG>",
            "detections": [...],
            "total_faces": int,
            "fps": float,
            "timestamp": "ISO 8601"
        }

        Status messages:
        {"type": "status", "message": "...", "status": "ok|error"}

        Snapshot:
        {"type": "snapshot", "frame": "<base64 JPEG>", "detections": [...]}
    ─────────────────────────────────────────────────────
    """
    await websocket.accept()
    
    stream = None
    streaming = False
    conf_threshold = detector.conf_threshold
    prev_time = time.time()
    
    try:
        # Gửi thông báo kết nối thành công
        await websocket.send_json({
            "type": "status",
            "message": "WebSocket connected. Send {\"action\": \"start\"} to begin streaming.",
            "status": "ok"
        })
        
        while True:
            # ── Kiểm tra có message từ client không (non-blocking khi đang stream) ──
            if streaming:
                try:
                    # Non-blocking receive với timeout ngắn
                    raw = await asyncio.wait_for(
                        websocket.receive_text(), 
                        timeout=0.005  # 5ms timeout — không block stream
                    )
                    command = json.loads(raw)
                    action = command.get("action", "")
                    
                    if action == "stop":
                        streaming = False
                        if stream:
                            stream.stop()
                            stream = None
                        await websocket.send_json({
                            "type": "status",
                            "message": "Stream stopped.",
                            "status": "ok"
                        })
                        continue
                    
                    elif action == "config":
                        if "conf_threshold" in command:
                            conf_threshold = float(command["conf_threshold"])
                            detector.conf_threshold = conf_threshold
                            await websocket.send_json({
                                "type": "status",
                                "message": f"Config updated: conf_threshold={conf_threshold}",
                                "status": "ok"
                            })
                    
                    elif action == "snapshot":
                        if stream and stream.is_opened():
                            ret, frame = stream.read()
                            if ret and frame is not None:
                                annotated, detections = detector.detect(frame)
                                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                b64 = base64.b64encode(buffer).decode("utf-8")
                                await websocket.send_json({
                                    "type": "snapshot",
                                    "frame": b64,
                                    "detections": _format_detections(detections),
                                    "total_faces": len(detections),
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                except asyncio.TimeoutError:
                    pass  # Không có message → tiếp tục stream
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "status",
                        "message": "Invalid JSON command.",
                        "status": "error"
                    })
                
                # ── Đọc frame và gửi về client ──
                if stream and stream.is_opened():
                    ret, frame = stream.read()
                    if ret and frame is not None:
                        # Detect faces
                        annotated, detections = detector.detect(frame)
                        
                        # Tính FPS
                        fps, prev_time = calculate_fps(prev_time)
                        
                        # Encode frame → base64 (cv2.imencode expects BGR)
                        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        b64 = base64.b64encode(buffer).decode("utf-8")
                        
                        # Gửi frame + detections về client
                        await websocket.send_json({
                            "type": "frame",
                            "frame": b64,
                            "detections": _format_detections(detections),
                            "total_faces": len(detections),
                            "fps": round(fps, 1),
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    # Điều khiển FPS (~25-30 FPS)
                    await asyncio.sleep(0.033)
                else:
                    streaming = False
                    await websocket.send_json({
                        "type": "status",
                        "message": "Camera disconnected.",
                        "status": "error"
                    })
            
            else:
                # ── Không đang stream → blocking receive chờ lệnh ──
                raw = await websocket.receive_text()
                command = json.loads(raw)
                action = command.get("action", "")
                
                if action == "start":
                    camera_index = command.get("camera_index", 0)
                    try:
                        stream = VideoStream(source=camera_index)
                        stream.start()
                        streaming = True
                        prev_time = time.time()
                        await websocket.send_json({
                            "type": "status",
                            "message": f"Stream started (camera {camera_index}).",
                            "status": "ok"
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "status",
                            "message": f"Failed to open camera: {str(e)}",
                            "status": "error"
                        })
                
                elif action == "snapshot":
                    # Snapshot 1 lần không cần stream liên tục
                    camera_index = command.get("camera_index", 0)
                    frame = _capture_single_frame(camera_index)
                    if frame is not None:
                        annotated, detections = detector.detect(frame)
                        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        b64 = base64.b64encode(buffer).decode("utf-8")
                        await websocket.send_json({
                            "type": "snapshot",
                            "frame": b64,
                            "detections": _format_detections(detections),
                            "total_faces": len(detections),
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        await websocket.send_json({
                            "type": "status",
                            "message": "Failed to capture snapshot.",
                            "status": "error"
                        })
                        
    except WebSocketDisconnect:
        pass  # Client ngắt kết nối — cleanup bên dưới
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "status",
                "message": f"Server error: {str(e)}",
                "status": "error"
            })
        except Exception:
            pass
    finally:
        # ── Cleanup: đảm bảo camera được release ──
        if stream:
            stream.stop()


@router.post(
    "/snapshot",
    summary="Chụp snapshot từ camera và nhận diện khuôn mặt",
)
async def realtime_snapshot(camera_index: int = 0):
    """
    Chụp 1 frame từ camera, detect khuôn mặt, trả về JSON.
    Không cần WebSocket — dùng cho nút SNAPSHOT trên giao diện.
    """
    frame = _capture_single_frame(camera_index)
    if frame is None:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Failed to capture from camera."}
        )
    
    _, detections = detector.detect(frame)
    
    return {
        "success": True,
        "total_faces": len(detections),
        "detections": _format_detections(detections),
        "timestamp": datetime.now().isoformat(),
        "message": f"Snapshot captured. Detected {len(detections)} face(s)."
    }


# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────

def _format_detections(detections: list) -> list:
    """Format detections cho JSON response."""
    return [
        {
            "bbox": {
                "x1": d["bbox"][0],
                "y1": d["bbox"][1],
                "x2": d["bbox"][2],
                "y2": d["bbox"][3],
            },
            "confidence": round(d["confidence"], 4),
            "class_id": d["class_id"],
        }
        for d in detections
    ]


def _capture_single_frame(camera_index: int = 0) -> np.ndarray | None:
    """Mở camera, chụp 1 frame, đóng camera ngay."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    
    # Đọc vài frame để camera ổn định (tránh frame đen)
    for _ in range(5):
        cap.read()
    
    ret, frame = cap.read()
    cap.release()
    
    return frame if ret else None
