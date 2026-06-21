"""
Canonical VideoRecord ORM model.

Owned by the shared package so every service (ingestion, preprocessing,
search …) works with the exact same table definition.  Alembic migrations
in infra/migrations/ manage schema changes against this model.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Single declarative base shared across the whole platform."""
    pass



class VideoStatus(str, enum.Enum):
    PENDING      = "PENDING"
    PROCESSING   = "PROCESSING"
    PREPROCESSED = "PREPROCESSED"   # add this line
    INDEXED      = "INDEXED"
    FAILED       = "FAILED"
    QUARANTINED  = "QUARANTINED"
    DUPLICATE    = "DUPLICATE"


class VideoRecord(Base):
    """One ingested video file.  Written by ingestion, read by every other service."""

    __tablename__ = "video_records"
    __table_args__ = (
        UniqueConstraint("sha256_hash", name="uq_video_hash"),
        Index("ix_video_status",     "status"),
        Index("ix_video_camera_id",  "camera_id"),
        Index("ix_video_created_at", "created_at"),
    )

    # ── Identity ──────────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    sha256_hash:       Mapped[str]       = mapped_column(String(64),   nullable=False, unique=True)
    original_filename: Mapped[str]       = mapped_column(String(512),  nullable=False)
    mime_type:         Mapped[str]       = mapped_column(String(128),  nullable=False)
    file_size_bytes:   Mapped[int]       = mapped_column(BigInteger,   nullable=False)

    # ── Storage ───────────────────────────────────────────────────────────────
    storage_path:   Mapped[str | None]  = mapped_column(String(1024), nullable=True)
    storage_bucket: Mapped[str | None]  = mapped_column(String(256),  nullable=True)

    # ── Camera / location metadata ────────────────────────────────────────────
    camera_id:          Mapped[str | None]      = mapped_column(String(256),            nullable=True)
    location:           Mapped[str | None]      = mapped_column(String(512),            nullable=True)
    recorded_at:        Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds:   Mapped[float | None]    = mapped_column(Float,                  nullable=True)
    resolution_width:   Mapped[int | None]      = mapped_column(Integer,                nullable=True)
    resolution_height:  Mapped[int | None]      = mapped_column(Integer,                nullable=True)

    # ── Pipeline state ────────────────────────────────────────────────────────
    status:        Mapped[VideoStatus] = mapped_column(
        Enum(VideoStatus), nullable=False, default=VideoStatus.PENDING
    )
    error_message: Mapped[str | None]  = mapped_column(Text, nullable=True)

    # ── Audit ─────────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<VideoRecord id={self.id} status={self.status} file={self.original_filename}>"