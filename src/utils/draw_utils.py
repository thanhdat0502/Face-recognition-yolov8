# Hàm hỗ trợ vẽ khung (bounding box)
import cv2
import numpy as np

def draw_bounding_boxes(image: np.ndarray, detections: list) -> np.ndarray:
    output = image.copy()
    color = (0, 255, 0)  # Màu xanh lá cây cho bounding box
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        confidence = det["confidence"]
        
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"face {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(output, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    return output

