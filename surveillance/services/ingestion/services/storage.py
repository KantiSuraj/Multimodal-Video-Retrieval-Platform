"""
Ingestion storage service.

Wraps shared.storage.ObjectStorageClient with ingestion-specific bucket
names and a quarantine helper.  No retry logic here — that lives in the
shared client.
"""
from __future__ import annotations

from uuid import UUID

from shared.storage.client import ObjectStorageClient
from services.ingestion.core.config import get_settings
from services.ingestion.core.logging import get_logger

logger   = get_logger(__name__)
settings = get_settings()


class IngestionStorageService:

    def __init__(self) -> None:
        self._client = ObjectStorageClient(settings)

    async def startup(self) -> None:
        """Ensure both buckets exist on MinIO."""
        for bucket in (settings.MINIO_RAW_BUCKET, settings.MINIO_QUARANTINE_BUCKET):
            await self._client.ensure_bucket(bucket)
        logger.info("storage_ready",
                    raw=settings.MINIO_RAW_BUCKET,
                    quarantine=settings.MINIO_QUARANTINE_BUCKET)

    async def upload_video(
        self,
        video_id:     UUID,
        data:         bytes,
        filename:     str,
        content_type: str,
    ) -> str:
        """Upload raw video; return object path."""
        object_name = f"{video_id}/{filename}"
        logger.info("uploading_video", video_id=str(video_id), size=len(data))
        path = await self._client.put_object(
            settings.MINIO_RAW_BUCKET, object_name, data, content_type
        )
        logger.info("upload_complete", video_id=str(video_id), path=path)
        return path

    async def quarantine(self, video_id: UUID, data: bytes, filename: str) -> str:
        """Store corrupt / rejected file in the quarantine bucket."""
        object_name = f"{video_id}/{filename}"
        logger.warning("quarantining_file", video_id=str(video_id), filename=filename)
        return await self._client.put_object(
            settings.MINIO_QUARANTINE_BUCKET, object_name, data, "application/octet-stream"
        )

    async def delete_object(self, object_name: str) -> None:
        """Best-effort cleanup of a partially uploaded object."""
        await self._client.delete_object(settings.MINIO_RAW_BUCKET, object_name)
        logger.info("partial_upload_deleted", path=object_name)


# Singleton — initialised in app lifespan
storage_service = IngestionStorageService()