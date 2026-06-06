import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from src.utils.configs import Settings

class FaceDetector:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        settings = Settings()
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {settings.MODEL_PATH}")
        self.model = YOLO(settings.MODEL_PATH)
        self.conf_threshold = settings.CONF_THRESHOLD
        self.imgsz = settings.INFER_IMGSZ
        
    def detect(self, image: np.ndarray):
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False
        )[0]
        
        annotated = image.copy()
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Ve bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Face {conf:.2f}"
            cv2.putText(
                annotated, label,
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id
            })
        return annotated, detections