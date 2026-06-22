import uuid

import pytest
from unittest.mock import AsyncMock, MagicMock

from shared.shared.events.frame_extracted import (
    ExtractedFrame,
    FramesExtractedEvent,
    PreprocessingMetadata,
)
from shared.shared.models.video import VideoRecord, VideoStatus

from services.detection.core.config import Settings
from services.detection.models.schemas import (
    DetectionError,
    DetectionStage,
    FrameRef,
    RawDetection,
)
from services.detection.services.detection import DetectionService


def _settings() -> Settings:
    return Settings(DETECTION_CONFIDENCE_THRESHOLD=0.4)


def _frames_event(video_id: uuid.UUID) -> FramesExtractedEvent:
    return FramesExtractedEvent(
        video_id=video_id,
        processed_video_path="normalized.mp4",
        processed_video_bucket="processed-videos",
        frames=[
            ExtractedFrame(
                frame_path=f"{video_id}/000000.jpg",
                sequence_index=0,
                timestamp_ms=0,
                scene_id=0,
                sharpness_score=120.0,
            )
        ],
        scenes=[],
        clips=[],
        preprocessing_metadata=PreprocessingMetadata(
            extraction_interval_seconds=1.0,
            blur_threshold=100.0,
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=8,
            scene_histogram_threshold=0.6,
            target_clip_duration_seconds=5,
            normalized_codec="h264",
            normalized_resolution="1280x720",
            normalized_fps=25,
        ),
    )


class TestGroundingDINODetectorConfidenceFiltering:
    @pytest.mark.asyncio
    async def test_low_confidence_detection_filtered_in_orchestrator(self, tmp_path):
        settings = _settings()
        detector = AsyncMock()
        detector.detect.return_value = [
            RawDetection("person", 0.9, 0.1, 0.1, 0.5, 0.5),
            RawDetection("car", 0.1, 0.0, 0.0, 0.2, 0.2),
        ]

        storage = AsyncMock()
        storage.fetch_frame.return_value = b"fake-bytes"

        publisher = AsyncMock()
        service = DetectionService(settings, detector, storage, publisher)

        video_id = uuid.uuid4()
        ref = FrameRef(frame_path="x/000000.jpg", sequence_index=0, timestamp_ms=0, scene_id=0)

        local_path = tmp_path / "000000.jpg"
        from PIL import Image

        Image.new("RGB", (100, 100)).save(local_path)

        async def fake_fetch_and_write(*args, **kwargs):
            return str(local_path)

        service._fetch_and_write_frame = fake_fetch_and_write  # type: ignore[assignment]

        async def fake_write_rows(*args, **kwargs):
            return None

        service._write_detection_rows = fake_write_rows  # type: ignore[assignment]

        results = await service._run_detection(video_id, _frames_event(video_id))
        kept_labels = [d.raw.label for d in results[0].detections]

        assert "person" in kept_labels
        assert "car" not in kept_labels


class TestDetectionServiceIdempotency:
    @pytest.mark.asyncio
    async def test_already_detected_video_is_skipped(self, monkeypatch):
        settings = _settings()
        detector = AsyncMock()
        storage = AsyncMock()
        publisher = AsyncMock()
        service = DetectionService(settings, detector, storage, publisher)

        record = MagicMock(spec=VideoRecord)
        record.status = VideoStatus.DETECTED

        async def fake_already_processed(video_id):
            return True

        service._already_processed = fake_already_processed  # type: ignore[assignment]

        video_id = uuid.uuid4()
        await service.process_frames(_frames_event(video_id))

        storage.fetch_frame.assert_not_called()
        publisher.publish_detection_complete.assert_not_called()


class TestDetectionServiceFailureModes:
    @pytest.mark.asyncio
    async def test_recoverable_fetch_failure_propagates_for_requeue(self):
        settings = _settings()
        detector = AsyncMock()
        storage = AsyncMock()
        storage.fetch_frame.side_effect = ConnectionError("minio down")
        publisher = AsyncMock()
        service = DetectionService(settings, detector, storage, publisher)

        async def fake_already_processed(video_id):
            return False

        async def fake_mark_status(*args, **kwargs):
            return None

        service._already_processed = fake_already_processed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]

        video_id = uuid.uuid4()
        with pytest.raises(DetectionError) as exc_info:
            await service.process_frames(_frames_event(video_id))

        assert exc_info.value.recoverable is True
        assert exc_info.value.stage == DetectionStage.FETCH_FRAMES

    @pytest.mark.asyncio
    async def test_non_recoverable_failure_quarantines_and_does_not_raise(self):
        settings = _settings()
        detector = AsyncMock()
        storage = AsyncMock()
        publisher = AsyncMock()
        service = DetectionService(settings, detector, storage, publisher)

        async def fake_already_processed(video_id):
            return False

        async def fake_mark_status(*args, **kwargs):
            return None

        async def fake_run_detection(video_id, event):
            raise DetectionError(
                message="model inference failed",
                stage=DetectionStage.MODEL_INFERENCE,
                recoverable=False,
            )

        service._already_processed = fake_already_processed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]
        service._run_detection = fake_run_detection  # type: ignore[assignment]

        video_id = uuid.uuid4()
        await service.process_frames(_frames_event(video_id))  # must not raise

        storage.quarantine.assert_called_once()


class TestDetectionStorageService:
    @pytest.mark.asyncio
    async def test_upload_crop_key_includes_sequence_and_detection_id(self):
        from services.detection.services.storage import DetectionStorageService

        client = AsyncMock()
        settings = _settings()
        storage = DetectionStorageService(client, settings)

        video_id = uuid.uuid4()
        detection_id = uuid.uuid4()
        bucket, key = await storage.upload_crop(video_id, 5, detection_id, b"jpeg-bytes")

        assert bucket == settings.MINIO_DETECTION_CROPS_BUCKET
        assert str(video_id) in key
        assert "000005" in key
        assert str(detection_id) in key