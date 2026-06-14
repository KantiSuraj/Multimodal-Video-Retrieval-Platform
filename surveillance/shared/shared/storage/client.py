"""
Shared MinIO / S3-compatible object storage client.

All services that need to read or write objects import this.
The ingestion service writes raw video; preprocessing reads it and writes
clips/frames; the search service reads frames for preview URLs.

Retries are handled here via tenacity so callers never worry about them.
"""
from __future__ import annotations

import asyncio
import io
from uuid import UUID

from minio import Minio
from minio.error import S3Error
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from shared.config.base import BaseServiceSettings


class ObjectStorageClient:
    """Async wrapper around the synchronous MinIO client."""

    def __init__(self, settings: BaseServiceSettings) -> None:
        self._settings = settings
        self._client   = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def ensure_bucket(self, name: str) -> None:
        """Create bucket if it doesn't already exist."""
        exists = await self._run(self._client.bucket_exists, name)
        if not exists:
            await self._run(self._client.make_bucket, name)

    # ── Write ─────────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((S3Error, ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def put_object(
        self,
        bucket:       str,
        object_name:  str,
        data:         bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload bytes; return the object_name (usable as a storage path)."""
        await self._run(
            self._client.put_object,
            bucket,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
        )
        return object_name

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_object(self, bucket: str, object_name: str) -> bytes:
        """Download and return the full object as bytes."""
        response = await self._run(self._client.get_object, bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete_object(self, bucket: str, object_name: str) -> None:
        try:
            await self._run(self._client.remove_object, bucket, object_name)
        except S3Error:
            pass  # best-effort cleanup

    # ── Presigned URL (for dashboard preview frames) ─────────────────────────

    async def presigned_get_url(
        self,
        bucket:      str,
        object_name: str,
        expires_sec: int = 3600,
    ) -> str:
        from datetime import timedelta
        return await self._run(
            self._client.presigned_get_object,
            bucket,
            object_name,
            expires=timedelta(seconds=expires_sec),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))