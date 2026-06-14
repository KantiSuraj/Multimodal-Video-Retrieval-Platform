"""
FramesExtractedEvent — published by preprocessing, consumed by detection.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FramesExtractedEvent(BaseModel):
    """Published to RabbitMQ routing key: video.frames_extracted"""

    event_type:     str      = "FramesExtractedEvent"
    video_id:       str
    clip_paths:     list[str]   # MinIO paths to 5-second clip files
    keyframe_paths: list[str]   # MinIO paths to individual keyframe images
    storage_bucket: str
    fps:            float
    duration_seconds: float
    resolution_width:  int
    resolution_height: int
    occurred_at:    datetime = Field(default_factory=datetime.utcnow)