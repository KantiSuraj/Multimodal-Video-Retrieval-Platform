from __future__ import annotations

import uuid

from shared.shared.storage.client import ObjectStorageClient

from services.detection.core.config import Settings


class DetectionStorageService:
    def __init__(self, client: ObjectStorageClient, settings: Settings):
        self._client = client
        self._settings = settings

    async def fetch_frame(self, bucket: str, path: str) -> bytes:
        return await self._client.get_object(bucket, path)

    async def upload_crop(
        self, video_id: uuid.UUID, sequence_index: int, detection_id: uuid.UUID, data: bytes
    ) -> tuple[str, str]:
        bucket = self._settings.MINIO_DETECTION_CROPS_BUCKET
        key = f"{video_id}/{sequence_index:06d}_{detection_id}.jpg"
        await self._client.put_object(bucket, key, data)
        return bucket, key

    async def quarantine(self, video_id: uuid.UUID, frame_path: str, data: bytes) -> None:
        bucket = self._settings.MINIO_QUARANTINE_DETECTION_BUCKET
        key = f"{video_id}/{frame_path.split('/')[-1]}"
        await self._client.put_object(bucket, key, data)