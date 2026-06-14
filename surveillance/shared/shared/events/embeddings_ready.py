"""
EmbeddingsReadyEvent — published by embedding, consumed by indexing.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EmbeddingRecord(BaseModel):
    """One embedding vector with its source context."""
    kind:         str           # "frame" | "crop" | "clip"
    source_path:  str           # MinIO path of the source image/clip
    vector:       list[float]   # unit-L2-normalised; length = model dim
    timestamp_ms: int | None = None   # position in the original video
    label:        str | None = None   # detection label (crop embeddings only)


class EmbeddingsReadyEvent(BaseModel):
    """Published to RabbitMQ routing key: video.embeddings_ready"""

    event_type:  str                   = "EmbeddingsReadyEvent"
    video_id:    str
    model_name:  str                   # e.g. "openai/clip-vit-large-patch14"
    embeddings:  list[EmbeddingRecord]
    occurred_at: datetime              = Field(default_factory=datetime.utcnow)