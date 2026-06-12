"""
Object storage service wrapping the MinIO client.

Features:
  - Async streaming upload with progress tracking
  - Automatic bucket creation on startup
  - Exponential-backoff retries (FR-ING-05 / failure matrix)
  - Separate quarantine bucket for corrupted files
"""
from __future__ import annotations

import asyncio
import io
from typing import AsyncIterator
from uuid import UUID

from minio import Minio
from minio.error import S3Error
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class StorageService:
    """Thin async wrapper around the synchronous MinIO client."""

    def __init__(self) -> None:
        self._client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """Ensure required buckets exist."""
        self._loop = asyncio.get_running_loop()
        for bucket in (settings.MINIO_RAW_BUCKET, settings.MINIO_QUARANTINE_BUCKET):
            await self._ensure_bucket(bucket)
        logger.info("storage_service_ready", buckets=[settings.MINIO_RAW_BUCKET, settings.MINIO_QUARANTINE_BUCKET])

    async def _ensure_bucket(self, name: str) -> None:
        exists = await self._run_sync(self._client.bucket_exists, name)
        if not exists:
            await self._run_sync(self._client.make_bucket, name)
            logger.info("bucket_created", bucket=name)

    # ── Upload ─────────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((S3Error, ConnectionError, TimeoutError)),
        stop=stop_after_attempt(settings.MINIO_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=settings.MINIO_RETRY_WAIT_SECONDS, min=1, max=10),
        reraise=True,
    )
    async def upload_video(
        self,
        video_id: UUID,
        data: bytes,
        filename: str,
        content_type: str,
        bucket: str | None = None,
    ) -> str:
        """Upload video bytes to MinIO; return the object path."""
        bucket = bucket or settings.MINIO_RAW_BUCKET
        object_name = f"{video_id}/{filename}"
        data_stream = io.BytesIO(data)

        logger.info(
            "uploading_video",
            video_id=str(video_id),
            bucket=bucket,
            object_name=object_name,
            size_bytes=len(data),
        )

        await self._run_sync(
            self._client.put_object,
            bucket,
            object_name,
            data_stream,
            len(data),
            content_type=content_type,
        )

        logger.info("upload_complete", video_id=str(video_id), path=object_name)
        return object_name

    async def upload_video_stream(
        self,
        video_id: UUID,
        stream: AsyncIterator[bytes],
        filename: str,
        content_type: str,
        total_size: int,
        bucket: str | None = None,
    ) -> str:
        """
        Stream-upload (chunked) for large files.
        Buffers into a BytesIO; swap for multipart upload in production
        if files regularly exceed available RAM.
        """
        bucket = bucket or settings.MINIO_RAW_BUCKET
        object_name = f"{video_id}/{filename}"
        buffer = io.BytesIO()
        received = 0

        async for chunk in stream:
            buffer.write(chunk)
            received += len(chunk)
            if total_size:
                pct = round(received / total_size * 100, 1)
                logger.debug("upload_progress", video_id=str(video_id), pct=pct)

        buffer.seek(0)
        await self._run_sync(
            self._client.put_object,
            bucket,
            object_name,
            buffer,
            received,
            content_type=content_type,
        )
        return object_name

    # ── Quarantine ────────────────────────────────────────────────────────────

    async def quarantine(self, video_id: UUID, data: bytes, filename: str) -> str:
        """Move corrupted/unsupported content to the quarantine bucket."""
        logger.warning("quarantining_file", video_id=str(video_id), filename=filename)
        return await self.upload_video(
            video_id=video_id,
            data=data,
            filename=filename,
            content_type="application/octet-stream",
            bucket=settings.MINIO_QUARANTINE_BUCKET,
        )

    async def delete_object(self, object_name: str, bucket: str | None = None) -> None:
        """Remove a partially-uploaded object on pipeline failure."""
        bucket = bucket or settings.MINIO_RAW_BUCKET
        try:
            await self._run_sync(self._client.remove_object, bucket, object_name)
            logger.info("partial_upload_deleted", bucket=bucket, object=object_name)
        except S3Error as exc:
            logger.warning("delete_failed", error=str(exc))

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _run_sync(self, fn, *args, **kwargs):
        """Run a blocking MinIO call in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


# Singleton instance (initialised in app lifespan)
storage_service = StorageService()
