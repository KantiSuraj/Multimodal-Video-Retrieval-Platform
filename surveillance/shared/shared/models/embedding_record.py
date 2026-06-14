"""
EmbeddingRecord ORM model.

Written by the embedding service, read by indexing and search.
One row per vector (frame-level, crop-level, or clip-level).
Actual vector bytes are stored in Qdrant; this table holds metadata
and a reference so we can re-index without re-embedding.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.models.video import Base


class EmbeddingRecord(Base):
    __tablename__ = "embedding_records"
    __table_args__ = (
        Index("ix_emb_video_id",  "video_id"),
        Index("ix_emb_kind",      "kind"),
        Index("ix_emb_qdrant_id", "qdrant_point_id"),
    )

    id:              Mapped[uuid.UUID]  = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id:        Mapped[uuid.UUID]  = mapped_column(UUID(as_uuid=True), ForeignKey("video_records.id"), nullable=False)

    # "frame" | "crop" | "clip"
    kind:            Mapped[str]        = mapped_column(String(16),   nullable=False)
    source_path:     Mapped[str]        = mapped_column(String(1024), nullable=False)
    model_name:      Mapped[str]        = mapped_column(String(256),  nullable=False)

    # Position in the video
    timestamp_ms:    Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Label from detection (crop embeddings only)
    label:           Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Reference to the vector in Qdrant
    qdrant_point_id: Mapped[str | None] = mapped_column(String(64),  nullable=True)
    qdrant_collection: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # Embedding dimensionality (sanity check on retrieval)
    vector_dim:      Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at:      Mapped[datetime]   = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )