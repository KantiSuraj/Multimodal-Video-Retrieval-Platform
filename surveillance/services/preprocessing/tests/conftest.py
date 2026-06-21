from __future__ import annotations

import uuid

import pytest

from services.preprocessing.core.config import Settings
from shared.shared.events.video_ingested import VideoIngestedEvent, VideoIngestedMetadata


@pytest.fixture
def settings() -> Settings:
    return Settings(TMP_DIR="/tmp/preprocessing_test")


@pytest.fixture
def video_ingested_event() -> VideoIngestedEvent:
    return VideoIngestedEvent(
        video_id=str(uuid.uuid4()),
        storage_path="abc/cam_east.mp4",
        storage_bucket="raw-videos",
        sha256_hash="a3f4b" * 12 + "abcd",
        original_filename="cam_east.mp4",
        mime_type="video/mp4",
        file_size_bytes=1024,
        metadata={"camera_id": "CAM-EAST", "duration_seconds": 120.5},
    )
