"""
services/preprocessing/core/config.py

Mirrors services/ingestion/core/config.py: a single Settings class that
inherits shared infra config and adds only what this service needs.
Every Stage 1-7 parameter mentioned in the spec is configurable via env
var with a sane default — nothing is hardcoded in the service modules.
"""
from __future__ import annotations

from functools import lru_cache

from shared.config.base import BaseServiceSettings


class Settings(BaseServiceSettings):
    # --- MinIO buckets owned by this service ---
    MINIO_RAW_BUCKET: str = "raw-videos"  # read-only source, owned by ingestion
    MINIO_PROCESSED_VIDEO_BUCKET: str = "processed-videos"
    MINIO_PROCESSED_FRAMES_BUCKET: str = "processed-frames"
    MINIO_PROCESSED_CLIPS_BUCKET: str = "processed-clips"
    MINIO_PREPROCESS_QUARANTINE_BUCKET: str = "quarantine-preprocessing"

    # --- RabbitMQ ---
    QUEUE_NAME: str = "preprocessing.tasks"
    CONSUME_ROUTING_KEY: str = "video.ingested"
    PUBLISH_ROUTING_KEY: str = "video.frames_extracted"

    # --- Stage 1: normalization ---
    NORMALIZED_CODEC: str = "libx264"
    NORMALIZED_RESOLUTION: str = "1280x720"
    NORMALIZED_FPS: int = 25
    FFMPEG_TIMEOUT_SECONDS: int = 1800

    # --- Stage 2: frame extraction ---
    FRAME_EXTRACTION_INTERVAL_SECONDS: float = 1.0

    # --- Stage 3: quality filtering ---
    BLUR_LAPLACIAN_VARIANCE_THRESHOLD: float = 100.0

    # --- Stage 4: CLAHE enhancement ---
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: int = 8

    # --- Stage 5: scene segmentation ---
    SCENE_HISTOGRAM_DIFF_THRESHOLD: float = 0.4

    # --- Stage 6: clip generation ---
    DEFAULT_CLIP_DURATION_SECONDS: float = 5.0
    MIN_CLIP_DURATION_SECONDS: float = 2.0
    MAX_CLIP_DURATION_SECONDS: float = 30.0

    # --- working directory for FFmpeg intermediates ---
    TMP_DIR: str = "/tmp/preprocessing"


@lru_cache
def get_settings() -> Settings:
    return Settings()
