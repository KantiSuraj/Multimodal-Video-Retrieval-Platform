"""
shared/events/frames_extracted.py

Published by: preprocessing
Consumed by: detection

DESIGN NOTE (see response to architect for full rationale):
This schema is being authored as part of the preprocessing service build —
the architecture doc lists this file as already existing but its shape was
never fixed. We define it now to the minimum surface detection needs:
enough to locate every keyframe it must run Grounding DINO on, plus scene/
clip boundaries so downstream services can group results without a second
round-trip to PostgreSQL. No detection/embedding-specific fields are added
here — only what preprocessing produces and detection consumes.
"""
from __future__ import annotations

import uuid
from typing import Literal

from datetime import datetime
from pydantic import Field
from pydantic import BaseModel


class ExtractedFrame(BaseModel):
    """One retained (post quality-filter, post-CLAHE) keyframe."""

    frame_path: str  # MinIO object key within `processed-frames`
    sequence_index: int  # 0-based order within the video
    timestamp_ms: int  # position in the *normalized* (25fps) video
    scene_id: int  # which segmented scene this frame belongs to
    sharpness_score: float  # Laplacian variance, kept for downstream debugging


class SceneBoundary(BaseModel):
    """One segmented scene, identified by histogram-difference thresholding."""

    scene_id: int
    start_ms: int
    end_ms: int


class GeneratedClip(BaseModel):
    """One generated clip covering a scene (or portion of a long scene)."""

    clip_path: str  # MinIO object key within `processed-clips`
    scene_id: int
    start_ms: int
    end_ms: int
    duration_seconds: float


class PreprocessingMetadata(BaseModel):
    """Parameters used for this run — lets detection/embedding reason about
    reproducibility, and lets an operator know what settings produced a
    given artifact set without re-deriving it from logs."""

    extraction_interval_seconds: float
    blur_threshold: float
    clahe_clip_limit: float
    clahe_tile_grid_size: int
    scene_histogram_threshold: float
    target_clip_duration_seconds: float
    normalized_codec: str
    normalized_resolution: str
    normalized_fps: int


class FramesExtractedEvent(BaseModel):
    event_type: Literal["FramesExtractedEvent"] = "FramesExtractedEvent"
    video_id: str
    processed_video_path: str  # transcoded H.264/720p/25fps MP4
    processed_video_bucket: str
    frames: list[ExtractedFrame]
    scenes: list[SceneBoundary]
    clips: list[GeneratedClip]
    preprocessing_metadata: PreprocessingMetadata
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
