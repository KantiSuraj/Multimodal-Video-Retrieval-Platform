"""
Base settings class.

Every service creates its own Settings that inherits from BaseServiceSettings
and adds service-specific fields.  Common infrastructure config (DB, MinIO,
RabbitMQ, Redis) lives here so it never drifts between services.

Usage in a service:
    from shared.config.base import BaseServiceSettings

    class Settings(BaseServiceSettings):
        MY_SERVICE_THING: str = "default"
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Identity ──────────────────────────────────────────────────────────────
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/surveillance"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # ── MinIO / S3 ────────────────────────────────────────────────────────────
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_RETRY_ATTEMPTS: int = 3
    MINIO_RETRY_WAIT_SECONDS: float = 1.0

    # ── RabbitMQ / AMQP ───────────────────────────────────────────────────────
    AMQP_URL: str = "amqp://guest:guest@localhost/"
    AMQP_EXCHANGE: str = "video.events"
    # Consumer tuning
    AMQP_PREFETCH_COUNT: int = 10

    # Future retry queue support
    AMQP_MAX_REDELIVERY_ATTEMPTS: int = 5

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"