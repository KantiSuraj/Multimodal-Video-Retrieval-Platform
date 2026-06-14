"""
Video ingestion REST API (FR-ING-01).

  POST /api/v1/videos          – multipart upload
  POST /api/v1/videos/rtsp     – RTSP pull
  POST /api/v1/videos/fs       – filesystem path
  GET  /api/v1/videos/{id}/status
"""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from services.ingestion.core.config  import get_settings
from services.ingestion.core.logging import get_logger
from services.ingestion.db.database  import get_db
from services.ingestion.models.schemas import (
    DuplicateVideoResponse,
    ErrorResponse,
    FilesystemIngestRequest,
    RTSPIngestRequest,
    VideoIngestResponse,
    VideoStatusResponse,
    VideoUploadMetadata,
)
from services.ingestion.services.ingestion import IngestionError, ingestion_service

logger   = get_logger(__name__)
settings = get_settings()
router   = APIRouter(prefix="/api/v1/videos", tags=["Video Ingestion"])


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=VideoIngestResponse | DuplicateVideoResponse,
    responses={
        200: {"model": DuplicateVideoResponse},
        202: {"model": VideoIngestResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Upload a video file (multipart/form-data)",
)
async def upload_video(
    file: UploadFile = File(...),
    metadata: str | None = Form(
        default=None,
        description='JSON: {"camera_id":"CAM-01","location":"Entrance","recorded_at":"2024-01-01T00:00:00Z"}',
    ),
    db: AsyncSession = Depends(get_db),
):
    data = await file.read(settings.MAX_UPLOAD_SIZE_BYTES + 1)
    if len(data) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_BYTES // (1024**3)} GB limit",
        )

    parsed_meta = VideoUploadMetadata()
    if metadata:
        try:
            parsed_meta = VideoUploadMetadata(**json.loads(metadata))
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}") from exc

    try:
        result = await ingestion_service.ingest_upload(
            db=db, file_data=data,
            filename=file.filename or "upload.mp4",
            metadata=parsed_meta,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return result


@router.post(
    "/rtsp",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=VideoIngestResponse,
    summary="Ingest a segment from an RTSP stream",
)
async def ingest_rtsp(body: RTSPIngestRequest, db: AsyncSession = Depends(get_db)):
    try:
        return await ingestion_service.ingest_rtsp(
            db=db, rtsp_url=body.rtsp_url,
            duration_seconds=body.duration_seconds,
            camera_id=body.camera_id, location=body.location,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "/fs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=VideoIngestResponse | DuplicateVideoResponse,
    summary="Ingest a video file from the server filesystem",
)
async def ingest_filesystem(body: FilesystemIngestRequest, db: AsyncSession = Depends(get_db)):
    try:
        return await ingestion_service.ingest_filesystem(
            db=db, file_path=body.file_path,
            camera_id=body.camera_id, location=body.location,
            recorded_at=body.recorded_at,
        )
    except (IngestionError, FileNotFoundError) as exc:
        code = exc.status_code if isinstance(exc, IngestionError) else 404  # type: ignore[attr-defined]
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.get(
    "/{video_id}/status",
    response_model=VideoStatusResponse,
    summary="Poll ingestion / processing status",
)
async def get_video_status(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    record = await ingestion_service.get_status(db, video_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return VideoStatusResponse.model_validate(record)