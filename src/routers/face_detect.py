# Chứa các API endpoint (ví dụ: /predict)
import io
import cv2
from fastapi import APIRouter, UploadFile, HTTPException, UploadFile, Query, File
from fastapi.responses import StreamingResponse, JSONResponse

from src.domain.face_detector import FaceDetector
from src.schemas.face_model import FaceDetectResponse, Detection, BoundingBox
from src.utils.image_utils import read_image_from_upload_file, encode_image_to_bytes

router = APIRouter(prefix="/face", tags=["Face Detection"])
detector = FaceDetector()

@router.post(
    "/detect/image",
    summary="Nhận diện khuôn mặt từ ảnh",
    response_class=StreamingResponse
)
async def detect_faces_in_image(
    file: UploadFile = File(..., description="Ảnh chứa khuôn mặt cần nhận diện"),
    format: str = Query("jpg", enum=["jpeg", "png"], description="Định dạng ảnh trả về (jpg, png)")
):
    image = await read_image_from_upload_file(file)
    annotated, _ = detector.detect(image)
    
    fmt = ".jpg" if format == "jpeg" else ".png"
    image_bytes = encode_image_to_bytes(annotated, fmt)
    
    return StreamingResponse(
        io.BytesIO(image_bytes), 
        media_type=f"image/{format}",
        headers={"X-Faces-Detected": str(len(_))}
    )
    
@router.post(
    "/detect/json",
    summary="Nhận diện khuôn mặt và trả về JSON",
    response_model=FaceDetectResponse
)
async def detect_faces_in_image_json(
    file: UploadFile = File(..., description="Ảnh chứa khuôn mặt cần nhận diện")
):
    image = await read_image_from_upload_file(file)
    _, detections = detector.detect(image)
    
    parsed = [
        Detection(
            bbox = BoundingBox(x1 = d["bbox"][0], y1=d["bbox"][1], x2=d["bbox"][2], y2=d["bbox"][3]),
            confidence = d["confidence"],
            class_id = d["class_id"]
        )
        for d in detections
    ]
    
    return FaceDetectResponse(
        success=True,
        total_faces=len(parsed),
        detections=parsed,
        message=f"Detected {len(parsed)} faces."
    )
    