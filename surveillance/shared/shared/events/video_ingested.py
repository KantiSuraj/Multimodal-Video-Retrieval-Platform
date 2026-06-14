"""
VideoIngestedEvent — published by the ingestion service, consumed by preprocessing.

This is the contract between those two teams.  Neither side should
define this schema independently; import from here.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class VideoIngestedEvent(BaseModel):
    """Published to RabbitMQ routing key: video.ingested"""

    event_type:        str  = "VideoIngestedEvent"
    video_id:          str
    storage_path:      str
    storage_bucket:    str
    sha256_hash:       str
    original_filename: str
    mime_type:         str
    file_size_bytes:   int
    metadata:          dict[str, Any] = Field(default_factory=dict)
    occurred_at:       datetime       = Field(default_factory=datetime.utcnow)