# File chính để chạy uvicorn server (FastAPI)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import face_detect
from src.utils.configs import Settings

settings = Settings()

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="API nhận diện khuôn mặt sử dụng YOLOv8",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(face_detect.router)

@app.get("/health", summary="Kiểm tra sức khỏe API")
def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}