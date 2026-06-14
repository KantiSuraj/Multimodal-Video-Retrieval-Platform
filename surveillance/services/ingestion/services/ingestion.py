"""
Ingestion orchestrator (FR-ING-01 → FR-ING-06).

Pipeline:
  1. Validate MIME type and extension
  2. Compute SHA-256 → check for duplicate
  3. Upload to MinIO (raw bucket)
  4. INSERT VideoRecord(status=PENDING) in PostgreSQL
  5. Publish VideoIngestedEvent → RabbitMQ
  6. Return 202 Accepted {video_id, polling_url}

Failure modes:
  MinIO unavailable  → 503  (after 3 retries in storage layer)
  DB write fails     → 500  + delete partial upload
  Duplicate          → 200  with existing video_id
  Corrupt / bad file → 422  + quarantine
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import aiohttp
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.events.video_ingested import VideoIngestedEvent
from shared.models.video import VideoRecord, VideoStatus

from services.ingestion.core.config import get_settings
from services.ingestion.core.logging import get_logger
from services.ingestion.models.schemas import (
    DuplicateVideoResponse,
    VideoIngestResponse,
    VideoUploadMetadata,
)
from services.ingestion.services.queue   import mq_publisher
from services.ingestion.services.storage import storage_service
from services.ingestion.services.validator import video_validator

logger   = get_logger(__name__)
settings = get_settings()


class IngestionError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class IngestionService:

    # ── Public entry points ───────────────────────────────────────────────────

    async def ingest_upload(
        self,
        db:        AsyncSession,
        file_data: bytes,
        filename:  str,
        metadata:  VideoUploadMetadata,
    ) -> VideoIngestResponse | DuplicateVideoResponse:
        return await self._run_pipeline(
            db=db, file_data=file_data, filename=filename,
            camera_id=metadata.camera_id,
            location=metadata.location,
            recorded_at=metadata.recorded_at,
        )

    async def ingest_rtsp(
        self,
        db:               AsyncSession,
        rtsp_url:         str,
        duration_seconds: int,
        camera_id:        str | None,
        location:         str | None,
    ) -> VideoIngestResponse:
        logger.info("rtsp_capture_start", url=rtsp_url, duration=duration_seconds)
        file_data, filename = await self._capture_rtsp(rtsp_url, duration_seconds)
        return await self._run_pipeline(  # type: ignore[return-value]
            db=db, file_data=file_data, filename=filename,
            camera_id=camera_id, location=location,
            recorded_at=datetime.utcnow(),
        )

    async def ingest_filesystem(
        self,
        db:          AsyncSession,
        file_path:   str,
        camera_id:   str | None,
        location:    str | None,
        recorded_at: datetime | None,
    ) -> VideoIngestResponse | DuplicateVideoResponse:
        import aiofiles
        logger.info("filesystem_ingest_start", path=file_path)
        async with aiofiles.open(file_path, "rb") as f:
            file_data = await f.read()
        return await self._run_pipeline(
            db=db, file_data=file_data,
            filename=file_path.split("/")[-1],
            camera_id=camera_id, location=location, recorded_at=recorded_at,
        )

    async def get_status(
        self, db: AsyncSession, video_id: uuid.UUID
    ) -> VideoRecord | None:
        result = await db.execute(
            select(VideoRecord).where(VideoRecord.id == video_id)
        )
        return result.scalar_one_or_none()

    # ── Core pipeline ─────────────────────────────────────────────────────────

    async def _run_pipeline(
        self,
        db:          AsyncSession,
        file_data:   bytes,
        filename:    str,
        camera_id:   str | None,
        location:    str | None,
        recorded_at: datetime | None,
    ) -> VideoIngestResponse | DuplicateVideoResponse:

        # Step 1 — validate
        validation = await video_validator.validate(file_data, filename)
        if not validation.is_valid:
            logger.warning("validation_failed", filename=filename, reason=validation.error_reason)
            await self._quarantine(db, uuid.uuid4(), file_data, filename,
                                   validation.error_reason or "Unknown")
            raise IngestionError(f"File validation failed: {validation.error_reason}", 422)

        # Step 2 — dedup
        existing = await self._find_by_hash(db, validation.sha256_hash)
        if existing is not None:
            logger.info("duplicate_detected", sha256=validation.sha256_hash,
                        existing_id=str(existing.id))
            return DuplicateVideoResponse(
                video_id=existing.id,
                status=existing.status.value,
                existing_storage_path=existing.storage_path,
            )

        video_id     = uuid.uuid4()
        storage_path = None

        # Step 3 — upload
        try:
            storage_path = await storage_service.upload_video(
                video_id=video_id, data=file_data,
                filename=filename, content_type=validation.mime_type,
            )
        except Exception as exc:
            logger.error("minio_upload_failed", video_id=str(video_id), error=str(exc))
            raise IngestionError("Object storage unavailable. Please retry.", 503) from exc

        # Step 4 — persist
        try:
            record = VideoRecord(
                id=video_id,
                sha256_hash=validation.sha256_hash,
                original_filename=filename,
                mime_type=validation.mime_type,
                file_size_bytes=len(file_data),
                storage_path=storage_path,
                storage_bucket=settings.MINIO_RAW_BUCKET,
                camera_id=camera_id,
                location=location,
                recorded_at=recorded_at,
                duration_seconds=validation.duration_seconds,
                resolution_width=validation.resolution_width,
                resolution_height=validation.resolution_height,
                status=VideoStatus.PENDING,
            )
            db.add(record)
            await db.flush()
        except Exception as exc:
            logger.error("db_write_failed", video_id=str(video_id), error=str(exc))
            if storage_path:
                await storage_service.delete_object(storage_path)
            raise IngestionError("Database write failed.", 500) from exc

        # Step 5 — publish event
        event = VideoIngestedEvent(
            video_id=str(video_id),
            storage_path=storage_path,
            storage_bucket=settings.MINIO_RAW_BUCKET,
            sha256_hash=validation.sha256_hash,
            original_filename=filename,
            mime_type=validation.mime_type,
            file_size_bytes=len(file_data),
            metadata={
                "camera_id":        camera_id,
                "location":         location,
                "recorded_at":      recorded_at.isoformat() if recorded_at else None,
                "duration_seconds": validation.duration_seconds,
                "resolution_width": validation.resolution_width,
                "resolution_height":validation.resolution_height,
                "codec":            validation.codec,
            },
        )
        try:
            await mq_publisher.publish_video_ingested(event)
        except Exception as exc:
            # Non-fatal: record already persisted; downstream can replay from DB
            logger.error("event_publish_failed", video_id=str(video_id), error=str(exc))

        logger.info("ingestion_accepted", video_id=str(video_id),
                    filename=filename, size=len(file_data))

        # Step 6 — respond
        return VideoIngestResponse(
            video_id=video_id,
            status=VideoStatus.PENDING.value,
            polling_url=f"/api/v1/videos/{video_id}/status",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    async def _find_by_hash(db: AsyncSession, sha256: str) -> VideoRecord | None:
        result = await db.execute(
            select(VideoRecord).where(VideoRecord.sha256_hash == sha256)
        )
        return result.scalar_one_or_none()

    async def _quarantine(
        self,
        db:       AsyncSession,
        video_id: uuid.UUID,
        data:     bytes,
        filename: str,
        reason:   str,
    ) -> None:
        qpath = None
        try:
            qpath = await storage_service.quarantine(video_id, data, filename)
        except Exception as exc:
            logger.error("quarantine_upload_failed", error=str(exc))

        record = VideoRecord(
            id=video_id,
            sha256_hash="",
            original_filename=filename,
            mime_type="application/octet-stream",
            file_size_bytes=len(data),
            storage_path=qpath,
            storage_bucket=settings.MINIO_QUARANTINE_BUCKET,
            status=VideoStatus.QUARANTINED,
            error_message=reason,
        )
        try:
            db.add(record)
            await db.flush()
        except Exception as exc:
            logger.error("quarantine_db_write_failed", error=str(exc))

        await self._fire_quarantine_webhook(str(video_id), filename, reason)

    @staticmethod
    async def _fire_quarantine_webhook(
        video_id: str, filename: str, reason: str
    ) -> None:
        url = settings.QUARANTINE_WEBHOOK_URL
        if not url:
            return
        payload: dict[str, Any] = {
            "event": "video.quarantined",
            "video_id": video_id,
            "filename": filename,
            "reason": reason,
        }
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(url, json=payload,
                                   timeout=aiohttp.ClientTimeout(total=5))
            logger.info("quarantine_webhook_fired", video_id=video_id)
        except Exception as exc:
            logger.warning("quarantine_webhook_failed", error=str(exc))

    @staticmethod
    async def _capture_rtsp(url: str, duration: int) -> tuple[bytes, str]:
        import os, tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            "ffmpeg", "-y", "-rtsp_transport", "tcp",
            "-i", url, "-t", str(duration), "-c", "copy", tmp_path,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            import asyncio as _asyncio
            _, stderr = await _asyncio.wait_for(proc.communicate(), timeout=duration + 30)
            if proc.returncode != 0:
                raise IngestionError(
                    f"RTSP capture failed: {stderr.decode(errors='replace')}", 422
                )
            import aiofiles
            async with aiofiles.open(tmp_path, "rb") as f:
                data = await f.read()
            return data, f"rtsp_capture_{uuid.uuid4().hex[:8]}.mp4"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


import asyncio  # noqa: E402 — needed by _capture_rtsp; keep at bottom to avoid circular

# Singleton
ingestion_service = IngestionService()