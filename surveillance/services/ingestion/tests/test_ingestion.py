"""
Unit tests for the ingestion service.

Run from the surveillance/ workspace root:
    pytest services/ingestion/tests/ -v
"""
from __future__ import annotations

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

FAKE_MP4_HEADER = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 100


@pytest.fixture
def fake_video_bytes():
    return FAKE_MP4_HEADER


@pytest.fixture
def fake_sha256(fake_video_bytes):
    return hashlib.sha256(fake_video_bytes).hexdigest()


# ── Validator ─────────────────────────────────────────────────────────────────

class TestVideoValidator:

    @pytest.mark.asyncio
    async def test_invalid_extension_rejected(self):
        from services.ingestion.services.validator import video_validator
        result = await video_validator.validate(b"data", "video.exe")
        assert not result.is_valid
        assert "extension" in (result.error_reason or "").lower()

    @pytest.mark.asyncio
    async def test_sha256_computed(self, fake_video_bytes, fake_sha256):
        from services.ingestion.services.validator import video_validator
        with (
            patch("magic.from_buffer", return_value="video/mp4"),
            patch.object(video_validator, "_ffprobe", new=AsyncMock(return_value={})),
        ):
            result = await video_validator.validate(fake_video_bytes, "test.mp4")
        assert result.sha256_hash == fake_sha256

    @pytest.mark.asyncio
    async def test_corrupt_file_fails(self):
        from services.ingestion.services.validator import video_validator
        with (
            patch("magic.from_buffer", return_value="video/mp4"),
            patch.object(video_validator, "_ffprobe", new=AsyncMock(return_value=None)),
        ):
            result = await video_validator.validate(b"corrupt", "video.mp4")
        assert not result.is_valid
        assert "corrupt" in (result.error_reason or "").lower()

    @pytest.mark.asyncio
    async def test_ffprobe_metadata_extracted(self, fake_video_bytes):
        from services.ingestion.services.validator import video_validator
        probe = {
            "format": {"duration": "120.5"},
            "streams": [{"codec_type": "video", "width": 1920, "height": 1080, "codec_name": "h264"}],
        }
        with (
            patch("magic.from_buffer", return_value="video/mp4"),
            patch.object(video_validator, "_ffprobe", new=AsyncMock(return_value=probe)),
        ):
            result = await video_validator.validate(fake_video_bytes, "cam.mp4")
        assert result.is_valid
        assert result.duration_seconds == pytest.approx(120.5)
        assert result.resolution_width  == 1920
        assert result.resolution_height == 1080
        assert result.codec == "h264"


# ── Ingestion service ─────────────────────────────────────────────────────────

class TestIngestionService:

    def _valid_result(self, sha256="abc123"):
        from services.ingestion.services.validator import ValidationResult
        return ValidationResult(
            is_valid=True, sha256_hash=sha256,
            mime_type="video/mp4", detected_extension=".mp4",
            duration_seconds=60.0, resolution_width=1280, resolution_height=720,
        )

    def _invalid_result(self):
        from services.ingestion.services.validator import ValidationResult
        return ValidationResult(is_valid=False, error_reason="corrupt")

    @pytest.mark.asyncio
    async def test_duplicate_returns_existing(self, fake_video_bytes, fake_sha256):
        from services.ingestion.services.ingestion import IngestionService
        from services.ingestion.models.schemas import DuplicateVideoResponse, VideoUploadMetadata
        from shared.shared.models.video import VideoRecord, VideoStatus

        existing = VideoRecord(
            id=uuid.uuid4(), sha256_hash=fake_sha256,
            original_filename="old.mp4", mime_type="video/mp4",
            file_size_bytes=100, status=VideoStatus.INDEXED,
            storage_path="abc/old.mp4",
        )
        svc = IngestionService()
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        with (
            patch("services.ingestion.services.ingestion.video_validator.validate",
                  new=AsyncMock(return_value=self._valid_result(sha256=fake_sha256))),
            patch.object(svc, "_find_by_hash", new=AsyncMock(return_value=existing)),
        ):
            result = await svc.ingest_upload(
                db=mock_db, file_data=fake_video_bytes,
                filename="test.mp4", metadata=VideoUploadMetadata(),
            )
        assert isinstance(result, DuplicateVideoResponse)
        assert result.video_id == existing.id

    @pytest.mark.asyncio
    async def test_invalid_file_quarantined_and_raises(self, fake_video_bytes):
        from services.ingestion.services.ingestion import IngestionService, IngestionError
        from services.ingestion.models.schemas import VideoUploadMetadata

        svc = IngestionService()
        with (
            patch("services.ingestion.services.ingestion.video_validator.validate",
                  new=AsyncMock(return_value=self._invalid_result())),
            patch.object(svc, "_quarantine", new=AsyncMock()),
        ):
            with pytest.raises(IngestionError) as exc_info:
                mock_db = AsyncMock()
                mock_db.add = MagicMock()
                await svc.ingest_upload(
                    db=mock_db, file_data=fake_video_bytes,
                    filename="bad.mp4", metadata=VideoUploadMetadata(),
                )
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_minio_failure_raises_503(self, fake_video_bytes, fake_sha256):
        from services.ingestion.services.ingestion import IngestionService, IngestionError
        from services.ingestion.models.schemas import VideoUploadMetadata

        svc = IngestionService()
        with (
            patch("services.ingestion.services.ingestion.video_validator.validate",
                  new=AsyncMock(return_value=self._valid_result(sha256=fake_sha256))),
            patch.object(svc, "_find_by_hash", new=AsyncMock(return_value=None)),
            patch("services.ingestion.services.ingestion.storage_service.upload_video",
                  new=AsyncMock(side_effect=ConnectionError("MinIO down"))),
        ):
            with pytest.raises(IngestionError) as exc_info:
                mock_db = AsyncMock()
                mock_db.add = MagicMock()
                await svc.ingest_upload(
                    db=mock_db, file_data=fake_video_bytes,
                    filename="test.mp4", metadata=VideoUploadMetadata(),
                )
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_db_failure_cleans_up_storage(self, fake_video_bytes, fake_sha256):
        from services.ingestion.services.ingestion import IngestionService, IngestionError
        from services.ingestion.models.schemas import VideoUploadMetadata

        svc      = IngestionService()
        mock_db  = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock(side_effect=Exception("DB down"))

        delete_mock = AsyncMock()
        with (
            patch("services.ingestion.services.ingestion.video_validator.validate",
                  new=AsyncMock(return_value=self._valid_result(sha256=fake_sha256))),
            patch.object(svc, "_find_by_hash", new=AsyncMock(return_value=None)),
            patch("services.ingestion.services.ingestion.storage_service.upload_video",
                  new=AsyncMock(return_value="abc/test.mp4")),
            patch("services.ingestion.services.ingestion.storage_service.delete_object",
                  delete_mock),
        ):
            with pytest.raises(IngestionError) as exc_info:
                await svc.ingest_upload(
                    db=mock_db, file_data=fake_video_bytes,
                    filename="test.mp4", metadata=VideoUploadMetadata(),
                )
        assert exc_info.value.status_code == 500
        delete_mock.assert_awaited_once_with("abc/test.mp4")

    @pytest.mark.asyncio
    async def test_successful_ingest_publishes_event(self, fake_video_bytes, fake_sha256):
        from services.ingestion.services.ingestion import IngestionService
        from services.ingestion.models.schemas import VideoIngestResponse, VideoUploadMetadata

        svc     = IngestionService()
        mock_db = AsyncMock()
        mock_db.add = MagicMock()

        with (
            patch("services.ingestion.services.ingestion.video_validator.validate",
                  new=AsyncMock(return_value=self._valid_result(sha256=fake_sha256))),
            patch.object(svc, "_find_by_hash", new=AsyncMock(return_value=None)),
            patch("services.ingestion.services.ingestion.storage_service.upload_video",
                  new=AsyncMock(return_value="abc/test.mp4")),
            patch("services.ingestion.services.ingestion.mq_publisher.publish_video_ingested",
                  new=AsyncMock()) as mock_pub,
        ):
            result = await svc.ingest_upload(
                db=mock_db, file_data=fake_video_bytes,
                filename="test.mp4", metadata=VideoUploadMetadata(camera_id="CAM-01"),
            )
        assert isinstance(result, VideoIngestResponse)
        assert result.status == "PENDING"
        mock_pub.assert_awaited_once()


# ── Storage service ───────────────────────────────────────────────────────────

class TestIngestionStorageService:

    @pytest.mark.asyncio
    async def test_upload_returns_object_path(self):
        from services.ingestion.services.storage import IngestionStorageService
        svc = IngestionStorageService()
        vid = uuid.uuid4()
        with patch.object(svc._client, "put_object", new=AsyncMock(return_value=f"{vid}/cam.mp4")):
            path = await svc.upload_video(vid, b"fake", "cam.mp4", "video/mp4")
        assert str(vid) in path
        assert "cam.mp4" in path

    @pytest.mark.asyncio
    async def test_quarantine_uses_quarantine_bucket(self):
        from services.ingestion.services.storage import IngestionStorageService
        from services.ingestion.core.config import get_settings
        svc = IngestionStorageService()
        vid = uuid.uuid4()
        calls = []
        async def fake_put(bucket, name, data, ct):
            calls.append(bucket)
            return name
        with patch.object(svc._client, "put_object", side_effect=fake_put):
            await svc.quarantine(vid, b"bad", "corrupt.mp4")
        assert calls[0] == get_settings().MINIO_QUARANTINE_BUCKET