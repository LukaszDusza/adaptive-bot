"""
Configuration management for Dashboard API
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Configuration
    API_TITLE: str = "Adaptive Bot Dashboard API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True  # Hot reload for development

    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",  # Vite default
        "http://localhost:5174",  # Vite alternative port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ]

    # Paths to bot data
    # Check if running in Docker (mounted volumes) or locally
    IN_DOCKER: bool = os.path.exists("/app/logs") and os.path.exists("/app/bot_state")

    PRICE_ACTION_DIR: str = "/app" if IN_DOCKER else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../price_action"))
    LOGS_DIR: str = "/app/logs" if IN_DOCKER else os.path.join(PRICE_ACTION_DIR, "logs")
    BOT_STATE_DIR: str = "/app/bot_state" if IN_DOCKER else os.path.join(PRICE_ACTION_DIR, "bot_state")
    MODELS_DIR: str = "/app/models" if IN_DOCKER else os.path.join(PRICE_ACTION_DIR, "models")
    PREDICTION_LOGS_DIR: str = "/app/prediction_logs" if IN_DOCKER else os.path.join(PRICE_ACTION_DIR, "prediction_logs")

    # Docker
    DOCKER_COMPOSE_FILE: str = os.path.join(PRICE_ACTION_DIR, "docker-compose.yaml")

    # Bybit API (for emergency close)
    BYBIT_API_KEY: Optional[str] = None
    BYBIT_API_SECRET: Optional[str] = None
    BYBIT_BASE_URL: str = "https://api.bybit.com"

    # WebSocket
    WS_PING_INTERVAL: int = 25
    WS_PING_TIMEOUT: int = 20

    # Cache settings
    CACHE_TTL: int = 10  # seconds

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), "../.env")
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()