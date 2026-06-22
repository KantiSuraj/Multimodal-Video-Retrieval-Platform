from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class DetectionStage(str, Enum):
    FETCH_FRAMES = "FETCH_FRAMES"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    CROP_PERSIST = "CROP_PERSIST"
    PUBLISH = "PUBLISH"


class DetectionError(Exception):
    def __init__(self, message: str, stage: DetectionStage, recoverable: bool):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.recoverable = recoverable


@dataclass
class FrameRef:
    frame_path: str
    sequence_index: int
    timestamp_ms: int
    scene_id: int


@dataclass
class RawDetection:
    label: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float


@dataclass
class PersistedDetection:
    detection_id: uuid.UUID
    raw: RawDetection
    crop_path: str | None  # bucket is implicit: settings.MINIO_DETECTION_CROPS_BUCKET


@dataclass
class PersistedFrameResult:
    frame: FrameRef
    detections: list[PersistedDetection] = field(default_factory=list)


@dataclass
class DetectionRunResult:
    frames: list[PersistedFrameResult] = field(default_factory=list)