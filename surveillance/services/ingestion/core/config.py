"""
Core configuration using pydantic-settings.
All values are read from environment variables with sane defaults.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "VideoIngestionService"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/surveillance"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # ── MinIO / S3-compatible object storage ─────────────────────────────────
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_RAW_BUCKET: str = "raw-videos"
    MINIO_QUARANTINE_BUCKET: str = "quarantine-videos"

    # ── RabbitMQ / AMQP ──────────────────────────────────────────────────────
    AMQP_URL: str = "amqp://guest:guest@localhost/"
    AMQP_EXCHANGE: str = "video.events"
    AMQP_ROUTING_KEY_INGESTED: str = "video.ingested"

    # ── Redis ────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Upload limits ────────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024 * 1024  # 10 GB
    ALLOWED_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".m2ts", ".mpeg"}
    ALLOWED_MIME_TYPES: set[str] = {
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
        "video/MP2T",
        "video/mpeg",
    }

    # ── Retry / resilience ────────────────────────────────────────────────────
    MINIO_RETRY_ATTEMPTS: int = 3
    MINIO_RETRY_WAIT_SECONDS: float = 1.0

    # ── Filesystem watcher ───────────────────────────────────────────────────
    WATCH_DIRECTORY: str = "/tmp/video_watch"
    WATCH_POLLING_INTERVAL: float = 2.0

    # ── Webhook for quarantine notifications ─────────────────────────────────
    QUARANTINE_WEBHOOK_URL: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()