from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IndexingStage(str, Enum):
    VALIDATE = "VALIDATE"
    COLLECTION_INIT = "COLLECTION_INIT"
    TRANSFORM = "TRANSFORM"
    UPSERT = "UPSERT"
    METADATA_PERSIST = "METADATA_PERSIST"
    STATUS_UPDATE = "STATUS_UPDATE"


class IndexingError(Exception):
    """Mirrors EmbeddingError: stage + recoverable flag for BaseConsumer."""

    def __init__(self, message: str, stage: IndexingStage, recoverable: bool):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.recoverable = recoverable


@dataclass
class QdrantPoint:
    """One vector ready for Qdrant upsert.

    point_id is a deterministic UUID5 derived from (video_id, source_path)
    so that duplicate event delivery updates the same point rather than
    creating duplicates.
    """

    point_id: str
    vector: list[float]
    payload: dict
