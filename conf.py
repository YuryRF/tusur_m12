from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Логи
    FILE_LOGGING: str = "logs/log.log"
    LOGGING_LEVEL: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"] = "INFO"

    # данные приложения
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_YOLO: Literal["yolov8n.pt", "yolov8s.pt", "yolov5s.pt"] = "yolov8s.pt"

    MAX_PHOTO_SIZE: int = 5
    IMG_TMP: str = 'img'


settings = Settings()

# end
