# Hàm hỗ trợ đọc stream video/webcam
import threading
import cv2
import time

class VideoStream:
    def __init__(self, source=0, resolution=(640, 480)):
        self.resource = source
        self.resolution = resolution
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self._lock = threading.Lock()
        self._thread = None
        
    def start(self):
        # Use V4L2 backend on Linux for reliable camera access
        self.cap = cv2.VideoCapture(self.resource, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            # Fallback: try without specifying backend
            self.cap = cv2.VideoCapture(self.resource)
        
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source {self.resource}")
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        time.sleep(1.0)
    
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret = ret
                self.frame = frame
    
    def read(self):
        with self._lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()
    
    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.frame = None
        
    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()
    
    def __del__(self):
        self.stop()
    
def calculate_fps(prev_time):
    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
    return fps, current_time

def draw_fps(frame, fps):
    cv2.putText(
        frame, f"FPS: {fps:.2f}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, (0, 255, 0), 2
    )
    return frame

