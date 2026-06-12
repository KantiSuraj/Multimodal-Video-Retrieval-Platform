"""
Test suite for the video ingestion service.

Run with:  pytest tests/ -v
"""
from __future__ import annotations

import hashlib
import io
import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_MP4_HEADER = (
    b"\x00\x00\x00\x1cftypisom"  # minimal ftyp box – recognized by libmagic as MP4
    + b"\x00" * 100
)


@pytest.fixture
def fake_video_bytes():
    return FAKE_MP4_HEADER


@pytest.fixture
def fake_sha256(fake_video_bytes):
    return hashlib.sha256(fake_video_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestVideoValidator:
    """Unit tests for the validator service."""

    @pytest.mark.asyncio
    async def test_invalid_extension_rejected(self):
        from services.validator import video_validator
        result = await video_validator.validate(b"data", "video.exe")
        assert not result.is_valid
        assert "extension" in (result.error_reason or "").lower()

    @pytest.mark.asyncio
    async def test_sha256_computed(self, fake_video_bytes):
        from services.validator import video_validator
        expected = hashlib.sha256(fake_video_bytes).hexdigest()
        with (
            patch("magic.from_buffer", return_value="video/mp4"),
            patch.object(video_validator, "_ffprobe", new=AsyncMock(return_value={})),
        ):
            result = await video_validator.validate(fake_video_bytes, "test.mp4")
        assert result.sha256_hash == expected

    @pytest.mark.asyncio
    async def test_corrupt_file_fails(self):
        from services.validator import video_validator
        with (
            patch("magic.from_buffer", return_value="video/mp4"),
            patch.object(video_validator, "_ffprobe", new=AsyncMock(return_value=None)),
        ):
            result = await video_validator.validate(b"corrupt", "video.mp4")
        assert not result.is_valid
        assert "corrupt" in (result.error_reason or "").lower()

    @pytest.mark.asyncio
    async def test_ffprobe_metadata_extracted(self, fake_video_bytes):
        from services.validator import video_validator
        probe = {
            "format": {"duration": "120.5"},
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080, "codec_name": "h264"}
            ],
        }
        with (
            patch("magic.from_buffer", return_value="video/mp4"),
            patch.object(video_validator, "_ffprobe", new=AsyncMock(return_value=probe)),
        ):
            result = await video_validator.validate(fake_video_bytes, "cam.mp4")
        assert result.is_valid
        assert result.duration_seconds == pytest.approx(120.5)
        assert result.resolution_width == 1920
        assert result.resolution_height == 1080
        assert result.codec == "h264"


# ---------------------------------------------------------------------------
# Ingestion service tests
# ---------------------------------------------------------------------------

class TestIngestionService:
    """Unit tests for the ingestion orchestrator."""

    def _make_validation_result(self, *, is_valid=True, sha256="abc123"):
        from services.validator import ValidationResult
        return ValidationResult(
            is_valid=is_valid,
            sha256_hash=sha256,
            mime_type="video/mp4",
            detected_extension=".mp4",
            duration_seconds=60.0,
            resolution_width=1280,
            resolution_height=720,
            error_reason=None if is_valid else "corrupt",
        )

    @pytest.mark.asyncio
    async def test_duplicate_returns_existing(self, fake_video_bytes, fake_sha256):
        from services.ingestion import IngestionService
        from models.video import VideoRecord, VideoStatus
        from models.schemas import VideoUploadMetadata

        existing = VideoRecord(
            id=uuid.uuid4(),
            sha256_hash=fake_sha256,
            original_filename="old.mp4",
            mime_type="video/mp4",
            file_size_bytes=100,
            status=VideoStatus.INDEXED,
            storage_path="abc/old.mp4",
        )

        svc = IngestionService()
        mock_db = AsyncMock()

        with (
            patch("services.ingestion.video_validator.validate", new=AsyncMock(
                return_value=self._make_validation_result(sha256=fake_sha256)
            )),
            patch.object(svc, "_find_by_hash", new=AsyncMock(return_value=existing)),
        ):
            result = await svc.ingest_upload(
                db=mock_db,
                file_data=fake_video_bytes,
                filename="test.mp4",
                metadata=VideoUploadMetadata(),
            )

        from models.schemas import DuplicateVideoResponse
        assert isinstance(result, DuplicateVideoResponse)
        assert result.video_id == existing.id

    @pytest.mark.asyncio
    async def test_invalid_file_raises_ingestion_error(self, fake_video_bytes):
        from services.ingestion import IngestionService, IngestionError
        from models.schemas import VideoUploadMetadata

        svc = IngestionService()
        mock_db = AsyncMock()

        with (
            patch("services.ingestion.video_validator.validate", new=AsyncMock(
                return_value=self._make_validation_result(is_valid=False)
            )),
            patch.object(svc, "_quarantine", new=AsyncMock()),
        ):
            with pytest.raises(IngestionError) as exc_info:
                await svc.ingest_upload(
                    db=mock_db,
                    file_data=fake_video_bytes,
                    filename="bad.mp4",
                    metadata=VideoUploadMetadata(),
                )
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_minio_failure_raises_503(self, fake_video_bytes):
        from services.ingestion import IngestionService, IngestionError
        from models.schemas import VideoUploadMetadata

        svc = IngestionService()
        mock_db = AsyncMock()

        with (
            patch("services.ingestion.video_validator.validate", new=AsyncMock(
                return_value=self._make_validation_result()
            )),
            patch.object(svc, "_find_by_hash", new=AsyncMock(return_value=None)),
            patch("services.ingestion.storage_service.upload_video", new=AsyncMock(
                side_effect=ConnectionError("MinIO down")
            )),
        ):
            with pytest.raises(IngestionError) as exc_info:
                await svc.ingest_upload(
                    db=mock_db,
                    file_data=fake_video_bytes,
                    filename="test.mp4",
                    metadata=VideoUploadMetadata(),
                )
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_successful_ingest_publishes_event(self, fake_video_bytes, fake_sha256):
        from services.ingestion import IngestionService
        from models.schemas import VideoIngestResponse, VideoUploadMetadata
        from models.video import VideoRecord, VideoStatus

        svc = IngestionService()
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        with (
            patch("services.ingestion.video_validator.validate", new=AsyncMock(
                return_value=self._make_validation_result(sha256=fake_sha256)
            )),
            patch.object(svc, "_find_by_hash", new=AsyncMock(return_value=None)),
            patch("services.ingestion.storage_service.upload_video", new=AsyncMock(
                return_value="abc123/test.mp4"
            )),
            patch("services.ingestion.mq_publisher.publish_video_ingested", new=AsyncMock()) as mock_pub,
        ):
            result = await svc.ingest_upload(
                db=mock_db,
                file_data=fake_video_bytes,
                filename="test.mp4",
                metadata=VideoUploadMetadata(camera_id="CAM-01"),
            )

        assert isinstance(result, VideoIngestResponse)
        assert result.status == "PENDING"
        mock_pub.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_db_failure_cleans_up_storage(self, fake_video_bytes, fake_sha256):
        from services.ingestion import IngestionService, IngestionError
        from models.schemas import VideoUploadMetadata

        svc = IngestionService()
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock(side_effect=Exception("DB down"))

        delete_mock = AsyncMock()
        with (
            patch("services.ingestion.video_validator.validate", new=AsyncMock(
                return_value=self._make_validation_result(sha256=fake_sha256)
            )),
            patch.object(svc, "_find_by_hash", new=AsyncMock(return_value=None)),
            patch("services.ingestion.storage_service.upload_video", new=AsyncMock(
                return_value="abc123/test.mp4"
            )),
            patch("services.ingestion.storage_service.delete_object", delete_mock),
        ):
            with pytest.raises(IngestionError) as exc_info:
                await svc.ingest_upload(
                    db=mock_db,
                    file_data=fake_video_bytes,
                    filename="test.mp4",
                    metadata=VideoUploadMetadata(),
                )
        assert exc_info.value.status_code == 500
        delete_mock.assert_awaited_once_with("abc123/test.mp4")


# ---------------------------------------------------------------------------
# Storage service tests
# ---------------------------------------------------------------------------

class TestStorageService:
    @pytest.mark.asyncio
    async def test_upload_returns_object_path(self):
        from services.storage import StorageService

        svc = StorageService()
        svc._client = MagicMock()
        svc._client.put_object = MagicMock()

        video_id = uuid.uuid4()
        with patch.object(svc, "_run_sync", new=AsyncMock(return_value=None)):
            path = await svc.upload_video(
                video_id=video_id,
                data=b"fake",
                filename="cam.mp4",
                content_type="video/mp4",
            )
        assert str(video_id) in path
        assert path.endswith("cam.mp4")

    @pytest.mark.asyncio
    async def test_quarantine_uses_quarantine_bucket(self):
        from services.storage import StorageService
        from core.config import get_settings

        svc = StorageService()
        upload_mock = AsyncMock(return_value="quarantine/abc.mp4")
        svc.upload_video = upload_mock

        video_id = uuid.uuid4()
        await svc.quarantine(video_id, b"bad data", "corrupt.mp4")

        upload_mock.assert_awaited_once()
        call_kwargs = upload_mock.call_args.kwargs
        assert call_kwargs["bucket"] == get_settings().MINIO_QUARANTINE_BUCKET
