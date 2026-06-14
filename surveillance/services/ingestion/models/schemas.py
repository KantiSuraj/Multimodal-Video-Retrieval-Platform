"""
Pydantic schemas local to the ingestion service.

Domain events (VideoIngestedEvent etc.) live in shared.events.
These schemas are HTTP API shapes — request bodies and response models —
so they belong to this service only.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ── Upload request metadata ───────────────────────────────────────────────────

class VideoUploadMetadata(BaseModel):
    camera_id:   str | None      = Field(None, max_length=256)
    location:    str | None      = Field(None, max_length=512)
    recorded_at: datetime | None = None


class RTSPIngestRequest(BaseModel):
    rtsp_url:         str         = Field(..., examples=["rtsp://192.168.1.100:554/stream1"])
    camera_id:        str | None  = None
    location:         str | None  = None
    duration_seconds: int         = Field(default=60, ge=5, le=3600)


class FilesystemIngestRequest(BaseModel):
    file_path:   str              = Field(..., description="Absolute path on the server filesystem")
    camera_id:   str | None      = None
    location:    str | None      = None
    recorded_at: datetime | None = None


# ── API responses ─────────────────────────────────────────────────────────────

class VideoIngestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    video_id:    uuid.UUID
    status:      str
    polling_url: str
    message:     str = "Video accepted for processing"


class DuplicateVideoResponse(BaseModel):
    video_id:              uuid.UUID
    status:                str
    message:               str       = "Duplicate: video already exists"
    existing_storage_path: str | None


class VideoStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:          uuid.UUID
    status:            str
    original_filename: str
    file_size_bytes:   int
    camera_id:         str | None
    location:          str | None
    recorded_at:       datetime | None
    duration_seconds:  float | None
    resolution_width:  int | None
    resolution_height: int | None
    storage_path:      str | None
    error_message:     str | None
    created_at:        datetime
    updated_at:        datetime


class ErrorResponse(BaseModel):
    detail: str
    code:   str | None = None