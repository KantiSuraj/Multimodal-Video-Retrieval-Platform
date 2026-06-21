"""
services/preprocessing/models/schemas.py

Internal pipeline DTOs, analogous to ingestion's ValidationResult. These
are dataclasses/pydantic models that pass data between stage modules
inside this service. None of these are persisted or published directly —
PreprocessingResult is translated into shared.events.frames_extracted at
the end of the pipeline.

PreprocessingError mirrors IngestionError(message, status_code): it carries
a `stage` (which step failed) and `recoverable` (should the message be
nack'd/retried, or does it indicate a structurally bad input that should be
quarantined instead). There is no HTTP status code here — this service has
no API layer, it's a pure queue consumer/worker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PreprocessingStage(str, Enum):
    FETCH_SOURCE = "FETCH_SOURCE"
    NORMALIZATION = "NORMALIZATION"
    FRAME_EXTRACTION = "FRAME_EXTRACTION"
    QUALITY_FILTER = "QUALITY_FILTER"
    ENHANCEMENT = "ENHANCEMENT"
    SCENE_SEGMENTATION = "SCENE_SEGMENTATION"
    CLIP_GENERATION = "CLIP_GENERATION"
    PERSISTENCE = "PERSISTENCE"
    PUBLISH = "PUBLISH"


class PreprocessingError(Exception):
    """Raised by any stage. `recoverable=True` means: a transient
    infrastructure problem (FFmpeg timeout, MinIO blip) — let the message
    be nack'd and redelivered. `recoverable=False` means: the input itself
    is bad (corrupt video, unreadable frames) — quarantine it and ack so it
    is not retried forever."""

    def __init__(self, message: str, stage: PreprocessingStage, recoverable: bool) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.recoverable = recoverable


@dataclass
class NormalizationResult:
    local_path: str
    width: int
    height: int
    fps: int
    duration_seconds: float
    codec: str


@dataclass
class FrameCandidate:
    sequence_index: int
    timestamp_ms: int
    local_path: str


@dataclass
class QualityFilterResult:
    kept: list[FrameCandidate]
    rejected_count: int


@dataclass
class EnhancedFrame:
    sequence_index: int
    timestamp_ms: int
    local_path: str
    sharpness_score: float


@dataclass
class SceneSegment:
    scene_id: int
    start_ms: int
    end_ms: int


@dataclass
class ClipSpec:
    scene_id: int
    start_ms: int
    end_ms: int
    local_path: str
    duration_seconds: float


@dataclass
class PreprocessingResult:
    """Aggregate of every stage's output, assembled by the orchestrator and
    handed to the storage layer (for persistence) then the queue layer
    (translated into FramesExtractedEvent)."""

    frames: list[EnhancedFrame] = field(default_factory=list)
    scenes: list[SceneSegment] = field(default_factory=list)
    clips: list[ClipSpec] = field(default_factory=list)
    normalized_video_local_path: str = ""
    normalized: NormalizationResult | None = None
