"""
Ingestion service settings.

Inherits all shared infrastructure config (DB, MinIO, AMQP, Redis)
from BaseServiceSettings and adds ingestion-specific fields only.
"""
from __future__ import annotations

from functools import lru_cache

from shared.config.base import BaseServiceSettings


class Settings(BaseServiceSettings):
    APP_NAME: str = "VideoIngestionService"

    # ── Buckets owned by this service ─────────────────────────────────────────
    MINIO_RAW_BUCKET:        str = "raw-videos"
    MINIO_QUARANTINE_BUCKET: str = "quarantine-videos"

    # ── Upload policy ─────────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024 * 1024   # 10 GB
    ALLOWED_EXTENSIONS: set[str] = {
        ".mp4", ".avi", ".mov", ".mkv", ".ts", ".m2ts", ".mpeg"
    }
    ALLOWED_MIME_TYPES: set[str] = {
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
        "video/MP2T",
        "video/mpeg",
    }

    # ── Routing keys this service publishes on ────────────────────────────────
    AMQP_ROUTING_KEY_INGESTED: str = "video.ingested"

    # ── Filesystem watcher ────────────────────────────────────────────────────
    WATCH_DIRECTORY: str = "/tmp/video_watch"

    # ── Quarantine webhook ────────────────────────────────────────────────────
    QUARANTINE_WEBHOOK_URL: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()