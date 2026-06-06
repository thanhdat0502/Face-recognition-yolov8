# Khai báo model Pydantic cho dữ liệu API
from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    
class Detection(BaseModel):
    bbox: BoundingBox
    confidence: float
    class_id: int
    
class FaceDetectResponse(BaseModel):
    success: bool
    total_faces: int
    detections: List[Detection]
    message: str = ""