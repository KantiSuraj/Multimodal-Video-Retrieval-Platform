"""
Pydantic v2 schemas for API request/response and internal events.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── Upload metadata (sent alongside the file) ─────────────────────────────────

class VideoUploadMetadata(BaseModel):
    """Optional metadata the client can supply at upload time."""
    camera_id: str | None = Field(None, max_length=256, examples=["CAM-EAST-01"])
    location: str | None = Field(None, max_length=512, examples=["Building A - Entrance"])
    recorded_at: datetime | None = Field(None, examples=["2024-06-01T08:30:00Z"])


class RTSPIngestRequest(BaseModel):
    """Body for the RTSP ingest endpoint."""
    rtsp_url: str = Field(..., examples=["rtsp://192.168.1.100:554/stream1"])
    camera_id: str | None = None
    location: str | None = None
    duration_seconds: int = Field(default=60, ge=5, le=3600)


class FilesystemIngestRequest(BaseModel):
    """Body for the local filesystem ingest endpoint."""
    file_path: str = Field(..., description="Absolute path to file on the server filesystem")
    camera_id: str | None = None
    location: str | None = None
    recorded_at: datetime | None = None


# ── API responses ─────────────────────────────────────────────────────────────

class VideoIngestResponse(BaseModel):
    """202 Accepted response returned after successful submission."""
    model_config = ConfigDict(from_attributes=True)

    video_id: uuid.UUID
    status: str
    polling_url: str
    message: str = "Video accepted for processing"


class VideoStatusResponse(BaseModel):
    """Response for status polling."""
    model_config = ConfigDict(from_attributes=True)

    video_id: uuid.UUID
    status: str
    original_filename: str
    file_size_bytes: int
    camera_id: str | None
    location: str | None
    recorded_at: datetime | None
    duration_seconds: float | None
    resolution_width: int | None
    resolution_height: int | None
    storage_path: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class DuplicateVideoResponse(BaseModel):
    """200 OK returned when an identical file has already been ingested."""
    video_id: uuid.UUID
    status: str
    message: str = "Duplicate: video already exists"
    existing_storage_path: str | None


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None


# ── Internal domain events ────────────────────────────────────────────────────

class VideoIngestedEvent(BaseModel):
    """Published to the message queue after successful ingestion."""
    event_type: str = "VideoIngestedEvent"
    video_id: str
    storage_path: str
    storage_bucket: str
    sha256_hash: str
    original_filename: str
    mime_type: str
    file_size_bytes: int
    metadata: dict[str, Any]
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
