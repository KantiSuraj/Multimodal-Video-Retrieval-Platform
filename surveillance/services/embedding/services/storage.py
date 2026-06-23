from __future__ import annotations

from shared.storage.client import ObjectStorageClient

from services.embedding.core.config import Settings


class EmbeddingStorageService:
    """Thin wrapper adding embedding-specific knowledge on top of the shared
    client. Embedding never writes artifacts back to MinIO — it only reads
    frames (from preprocessing's bucket) and crops (from detection's
    bucket), so there is no upload_*/quarantine() method here, unlike
    DetectionStorageService.
    """

    def __init__(self, client: ObjectStorageClient, settings: Settings):
        self._client = client
        self._settings = settings

    async def fetch_artifact(self, bucket: str, path: str) -> bytes:
        return await self._client.get_object(bucket, path)
