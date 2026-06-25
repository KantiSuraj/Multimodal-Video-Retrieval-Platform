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
    """Published to RabbitMQ routing key: video.embeddings_ready

    When the embedding service splits a video's embeddings across multiple
    RabbitMQ messages (to avoid frame-size overflow), each message carries:
      - batch_index   : 0-based position of this batch
      - total_batches : total number of batches for this video

    Indexing must NOT transition VideoRecord.status to INDEXED until it
    processes the batch where batch_index == total_batches - 1  (the last
    one).  All earlier batches are upserted to Qdrant and then acked
    without updating the video status.

    For backward-compatibility with single-batch scenarios both fields
    default to 0 and 1 respectively, which means "this is the only batch".
    """

    event_type:   str                   = "EmbeddingsReadyEvent"
    video_id:     str
    model_name:   str                   # e.g. "openai/clip-vit-large-patch14"
    embeddings:   list[EmbeddingRecord]
    occurred_at:  datetime              = Field(default_factory=datetime.utcnow)
    # Batch co-ordination fields — populated by the embedding service
    # when it splits a large video across multiple messages.
    batch_index:   int = 0  # 0-based index of this batch
    total_batches: int = 1  # total number of batches for this video