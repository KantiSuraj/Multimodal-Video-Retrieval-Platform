"""
services/preprocessing/services/storage.py

Mirrors services/ingestion/services/storage.py: wraps
shared.storage.ObjectStorageClient and adds this service's own knowledge —
bucket names and object path conventions. All retry logic stays in the
shared client; this file only knows *where* things go.

Path conventions (parallel to ingestion's `{video_id}/{filename}`):
  processed-videos/{video_id}/normalized.mp4
  processed-frames/{video_id}/{sequence_index:06d}.jpg
  processed-clips/{video_id}/scene_{scene_id:04d}.mp4
  quarantine-preprocessing/{video_id}/{original_filename}
"""
from __future__ import annotations

import uuid

from shared.storage.client import ObjectStorageClient

from services.preprocessing.core.config import Settings
from services.preprocessing.core.logging import get_logger

logger = get_logger(__name__)


class PreprocessingStorageService:
    def __init__(self, settings: Settings, storage_client: ObjectStorageClient) -> None:
        self._settings = settings
        self._client = storage_client

    async def ensure_buckets(self) -> None:
        for bucket in (
            self._settings.MINIO_PROCESSED_VIDEO_BUCKET,
            self._settings.MINIO_PROCESSED_FRAMES_BUCKET,
            self._settings.MINIO_PROCESSED_CLIPS_BUCKET,
            self._settings.MINIO_PREPROCESS_QUARANTINE_BUCKET,
        ):
            await self._client.ensure_bucket(bucket)

    async def fetch_raw_video(self, storage_bucket: str, storage_path: str) -> bytes:
        return await self._client.get_object(storage_bucket, storage_path)

    async def upload_processed_video(self, video_id: uuid.UUID, data: bytes) -> str:
        key = f"{video_id}/normalized.mp4"
        await self._client.put_object(
            self._settings.MINIO_PROCESSED_VIDEO_BUCKET, key, data, "video/mp4"
        )
        return key

    async def upload_frame(self, video_id: uuid.UUID, sequence_index: int, data: bytes) -> str:
        key = f"{video_id}/{sequence_index:06d}.jpg"
        await self._client.put_object(
            self._settings.MINIO_PROCESSED_FRAMES_BUCKET, key, data, "image/jpeg"
        )
        return key

    async def upload_clip(self, video_id: uuid.UUID, scene_id: int, data: bytes) -> str:
        key = f"{video_id}/scene_{scene_id:04d}.mp4"
        await self._client.put_object(
            self._settings.MINIO_PROCESSED_CLIPS_BUCKET, key, data, "video/mp4"
        )
        return key

    async def quarantine(self, video_id: uuid.UUID, filename: str, data: bytes) -> str:
        key = f"{video_id}/{filename}"
        await self._client.put_object(
            self._settings.MINIO_PREPROCESS_QUARANTINE_BUCKET,
            key,
            data,
            "application/octet-stream",
        )
        return key
