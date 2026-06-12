"""
Video ingestion API router.

Endpoints:
  POST /api/v1/videos          – multipart upload (FR-ING-01)
  POST /api/v1/videos/rtsp     – RTSP pull ingest
  POST /api/v1/videos/fs       – filesystem path ingest
  GET  /api/v1/videos/{id}/status – polling endpoint (FR-ING-04)
"""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.database import get_db
from core.logging import get_logger
from models.schemas import (
    DuplicateVideoResponse,
    ErrorResponse,
    FilesystemIngestRequest,
    RTSPIngestRequest,
    VideoIngestResponse,
    VideoStatusResponse,
    VideoUploadMetadata,
)
from services.ingestion import IngestionError, ingestion_service

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/v1/videos", tags=["Video Ingestion"])


# ── POST /api/v1/videos ───────────────────────────────────────────────────────

@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=VideoIngestResponse | DuplicateVideoResponse,
    responses={
        200: {"model": DuplicateVideoResponse, "description": "Duplicate – already ingested"},
        202: {"model": VideoIngestResponse, "description": "Accepted for processing"},
        422: {"model": ErrorResponse, "description": "Validation / corrupt file"},
        503: {"model": ErrorResponse, "description": "Storage unavailable"},
    },
    summary="Upload a video file (multipart/form-data)",
)
async def upload_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, MKV, MPEG-2 TS)"),
    metadata: str | None = Form(
        default=None,
        description='Optional JSON string: {"camera_id":"CAM-01","location":"Entrance","recorded_at":"2024-01-01T00:00:00Z"}',
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a video upload via multipart/form-data.

    - Validates MIME type and extension
    - Deduplicates by SHA-256 (returns 200 if duplicate)
    - Stores in MinIO and inserts a PENDING record in PostgreSQL
    - Publishes `VideoIngestedEvent` to the message queue
    - Returns 202 with `video_id` and polling URL
    """
    # Size guard – read up to MAX + 1 byte to detect over-limit
    data = await file.read(settings.MAX_UPLOAD_SIZE_BYTES + 1)
    if len(data) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024**3)} GB",
        )

    parsed_meta = VideoUploadMetadata()
    if metadata:
        try:
            parsed_meta = VideoUploadMetadata(**json.loads(metadata))
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}") from exc

    try:
        result = await ingestion_service.ingest_upload(
            db=db,
            file_data=data,
            filename=file.filename or "upload.mp4",
            metadata=parsed_meta,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    # 200 for duplicates, 202 for new uploads
    if isinstance(result, DuplicateVideoResponse):
        return result  # FastAPI returns 200 for this
    return result  # 202


# ── POST /api/v1/videos/rtsp ──────────────────────────────────────────────────

@router.post(
    "/rtsp",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=VideoIngestResponse,
    summary="Ingest a segment from an RTSP stream",
)
async def ingest_rtsp(
    body: RTSPIngestRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Pull `duration_seconds` of footage from an RTSP URL and run the
    standard ingestion pipeline.
    """
    try:
        result = await ingestion_service.ingest_rtsp(
            db=db,
            rtsp_url=body.rtsp_url,
            duration_seconds=body.duration_seconds,
            camera_id=body.camera_id,
            location=body.location,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return result


# ── POST /api/v1/videos/fs ────────────────────────────────────────────────────

@router.post(
    "/fs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=VideoIngestResponse | DuplicateVideoResponse,
    summary="Ingest a video file from the server filesystem",
)
async def ingest_filesystem(
    body: FilesystemIngestRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Ingest a video already present on the server's local filesystem.
    Useful for bulk imports or automated pipelines that drop files into
    a shared mount.
    """
    try:
        result = await ingestion_service.ingest_filesystem(
            db=db,
            file_path=body.file_path,
            camera_id=body.camera_id,
            location=body.location,
            recorded_at=body.recorded_at,
        )
    except (IngestionError, FileNotFoundError) as exc:
        code = exc.status_code if isinstance(exc, IngestionError) else 404  # type: ignore[attr-defined]
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    return result


# ── GET /api/v1/videos/{video_id}/status ─────────────────────────────────────

@router.get(
    "/{video_id}/status",
    response_model=VideoStatusResponse,
    summary="Poll ingestion / processing status for a video",
)
async def get_video_status(
    video_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Return the current status and metadata for a previously submitted video."""
    record = await ingestion_service.get_status(db, video_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return VideoStatusResponse.model_validate(record)
