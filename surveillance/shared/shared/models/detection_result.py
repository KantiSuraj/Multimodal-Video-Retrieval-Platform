"""
DetectionResult ORM model.

Written by the detection service, read by embedding and search.
One row per detected object per frame.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.models.video import Base


class DetectionResult(Base):
    __tablename__ = "detection_results"
    __table_args__ = (
        Index("ix_det_video_id",  "video_id"),
        Index("ix_det_frame_ts",  "frame_timestamp_ms"),
        Index("ix_det_scene_id",  "scene_id"),
    )

    id:                  Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id:            Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("video_records.id"), nullable=False)

    frame_path:          Mapped[str]       = mapped_column(String(1024), nullable=False)
    frame_timestamp_ms:  Mapped[int]       = mapped_column(Integer, nullable=False)
    scene_id:            Mapped[int]       = mapped_column(Integer, nullable=False, default=0)  # new — additive

    label:               Mapped[str]       = mapped_column(String(512), nullable=False)
    confidence:          Mapped[float]     = mapped_column(Float, nullable=False)

    bbox_x1:             Mapped[float]     = mapped_column(Float, nullable=False)
    bbox_y1:             Mapped[float]     = mapped_column(Float, nullable=False)
    bbox_x2:             Mapped[float]     = mapped_column(Float, nullable=False)
    bbox_y2:             Mapped[float]     = mapped_column(Float, nullable=False)

    crop_path:           Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at:          Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)