"""
DetectionCompleteEvent — published by detection, consumed by embedding.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    frame_path:  str
    label:       str
    confidence:  float
    bbox:        BoundingBox
    crop_path:   str | None = None   # MinIO path to the cropped sub-image
    extra:       dict[str, Any] = Field(default_factory=dict)


class DetectionCompleteEvent(BaseModel):
    """Published to RabbitMQ routing key: video.detection_complete"""

    event_type:  str             = "DetectionCompleteEvent"
    video_id:    str
    detections:  list[Detection]
    frame_count: int
    occurred_at: datetime        = Field(default_factory=datetime.utcnow)