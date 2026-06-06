from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    MODEL_PATH: str = "models/face_yolov8.pt"
    CONF_THRESHOLD: float = 0.4
    INFER_IMGSZ: int = 1280
    APP_TITLE: str = "Face Detection API"
    APP_VERSION: str = "1.0.0"
    MAX_IMAGE_SIZE_MB: int = 10
