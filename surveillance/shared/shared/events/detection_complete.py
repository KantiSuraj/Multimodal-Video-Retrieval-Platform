"""Event published by detection after a frame batch finishes a successful detection pass.

Consumed by: embedding service.
Mirrors the schema discipline of frames_extracted.py — additive fields only,
event_type is a fixed Literal discriminator, every coordinate downstream needs
to locate an artifact is carried in the payload so embedding never needs a
second lookup against detection's own tables.
"""
from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from shared.events.base import BaseEvent


class Detection(BaseModel):
    """One detected object instance within a single frame."""

    detection_id: uuid.UUID
    label: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    crop_path: str | None = None
    crop_bucket: str | None = None


class FrameDetections(BaseModel):
    """All detections found in one frame, plus the frame's own identity."""

    frame_path: str
    sequence_index: int
    timestamp_ms: int
    scene_id: int
    detections: list[Detection] = Field(default_factory=list)


class DetectionMetadata(BaseModel):
    """Configuration snapshot for this detection run — same reproducibility
    rationale as preprocessing_metadata in FramesExtractedEvent."""

    model_name: str
    text_prompt: str
    box_threshold: float
    text_threshold: float
    confidence_threshold: float


class DetectionCompleteEvent(BaseEvent):
    event_type: str = "DetectionCompleteEvent"
    video_id: uuid.UUID
    frames: list[FrameDetections] = Field(default_factory=list)
    detection_metadata: DetectionMetadata