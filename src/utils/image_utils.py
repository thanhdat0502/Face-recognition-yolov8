# Hàm hỗ trợ xử lý mảng/hình ảnh byte
import cv2
import numpy as np
from fastapi import UploadFile, HTTPException

MAX_SIZE_MB = 10

async def read_image_from_upload_file(file: UploadFile) -> np.ndarray:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    
    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_SIZE_MB} MB.")
    
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    
    return image

def encode_image_to_bytes(image: np.ndarray, fmt: str=".jpg") -> bytes:
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise ValueError("Could not encode image.")
    return buffer.tobytes()