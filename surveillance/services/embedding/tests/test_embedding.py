import uuid

import pytest
from unittest.mock import AsyncMock

from shared.events.detection_complete import (
    Detection,
    DetectionCompleteEvent,
    DetectionMetadata,
    FrameDetections,
)
from shared.models.video import VideoStatus

from services.embedding.core.config import Settings
from services.embedding.models.schemas import (
    ArtifactRef,
    EmbeddingError,
    EmbeddingStage,
)
from services.embedding.services.embedding import EmbeddingService


def _settings() -> Settings:
    return Settings()


def _detection_event(video_id: uuid.UUID) -> DetectionCompleteEvent:
    return DetectionCompleteEvent(
        video_id=video_id,
        frames=[
            FrameDetections(
                frame_path=f"{video_id}/000000.jpg",
                sequence_index=0,
                timestamp_ms=0,
                scene_id=0,
                detections=[
                    Detection(
                        detection_id=uuid.uuid4(),
                        label="person",
                        confidence=0.9,
                        bbox_x1=0.1,
                        bbox_y1=0.1,
                        bbox_x2=0.5,
                        bbox_y2=0.5,
                        crop_path=f"{video_id}/000000_crop0.jpg",
                        crop_bucket="detection-crops",
                    ),
                    Detection(
                        detection_id=uuid.uuid4(),
                        label="bag",
                        confidence=0.6,
                        bbox_x1=0.0,
                        bbox_y1=0.0,
                        bbox_x2=0.2,
                        bbox_y2=0.2,
                        crop_path=None,
                    ),
                ],
            )
        ],
        detection_metadata=DetectionMetadata(
            model_name="IDEA-Research/grounding-dino-tiny",
            text_prompt="person. car. backpack.",
            box_threshold=0.35,
            text_threshold=0.25,
            confidence_threshold=0.4,
        ),
    )


class TestEmbeddingServiceArtifactCollection:
    def test_collects_one_frame_and_only_crops_with_paths(self):
        settings = _settings()
        model = AsyncMock()
        storage = AsyncMock()
        publisher = AsyncMock()
        service = EmbeddingService(settings, model, storage, publisher)

        video_id = uuid.uuid4()
        artifacts = service._collect_artifacts(_detection_event(video_id))

        kinds = [a.kind for a in artifacts]
        assert kinds.count("frame") == 1
        # The "bag" detection has no crop_path — it must not produce an artifact.
        assert kinds.count("crop") == 1

        crop = next(a for a in artifacts if a.kind == "crop")
        assert crop.source_bucket == "detection-crops"
        assert crop.label == "person"

    def test_crop_falls_back_to_default_bucket_when_event_omits_it(self):
        settings = _settings()
        model = AsyncMock()
        storage = AsyncMock()
        publisher = AsyncMock()
        service = EmbeddingService(settings, model, storage, publisher)

        video_id = uuid.uuid4()
        event = _detection_event(video_id)
        event.frames[0].detections[0].crop_bucket = None

        artifacts = service._collect_artifacts(event)
        crop = next(a for a in artifacts if a.kind == "crop")

        assert crop.source_bucket == settings.MINIO_DETECTION_CROPS_BUCKET


class TestEmbeddingServiceIdempotency:
    @pytest.mark.asyncio
    async def test_already_embedded_video_is_skipped(self):
        settings = _settings()
        model = AsyncMock()
        storage = AsyncMock()
        publisher = AsyncMock()
        service = EmbeddingService(settings, model, storage, publisher)

        async def fake_already_processed(video_id):
            return True

        service._already_processed = fake_already_processed  # type: ignore[assignment]

        video_id = uuid.uuid4()
        await service.process_detections(_detection_event(video_id))

        storage.fetch_artifact.assert_not_called()
        publisher.publish_embeddings_ready.assert_not_called()


class TestEmbeddingServiceFailureModes:
    @pytest.mark.asyncio
    async def test_recoverable_fetch_failure_propagates_for_requeue(self):
        settings = _settings()
        model = AsyncMock()
        storage = AsyncMock()
        storage.fetch_artifact.side_effect = ConnectionError("minio down")
        publisher = AsyncMock()
        service = EmbeddingService(settings, model, storage, publisher)

        async def fake_already_processed(video_id):
            return False

        async def fake_mark_status(*args, **kwargs):
            return None

        async def fake_clear(*args, **kwargs):
            return None

        service._already_processed = fake_already_processed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]
        service._clear_existing_records = fake_clear  # type: ignore[assignment]

        video_id = uuid.uuid4()
        with pytest.raises(EmbeddingError) as exc_info:
            await service.process_detections(_detection_event(video_id))

        assert exc_info.value.recoverable is True
        assert exc_info.value.stage == EmbeddingStage.FETCH_ARTIFACT

    @pytest.mark.asyncio
    async def test_non_recoverable_failure_marks_failed_and_does_not_raise(self):
        settings = _settings()
        model = AsyncMock()
        storage = AsyncMock()
        publisher = AsyncMock()
        service = EmbeddingService(settings, model, storage, publisher)

        statuses: list[VideoStatus] = []

        async def fake_already_processed(video_id):
            return False

        async def fake_clear(*args, **kwargs):
            return None

        async def fake_mark_status(video_id, status, error_message=None):
            statuses.append(status)

        async def fake_embed_artifacts(video_id, artifacts):
            raise EmbeddingError(
                message="model inference failed",
                stage=EmbeddingStage.MODEL_INFERENCE,
                recoverable=False,
            )

        service._already_processed = fake_already_processed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]
        service._clear_existing_records = fake_clear  # type: ignore[assignment]
        service._embed_artifacts = fake_embed_artifacts  # type: ignore[assignment]

        video_id = uuid.uuid4()
        await service.process_detections(_detection_event(video_id))  # must not raise

        assert VideoStatus.FAILED in statuses
        assert VideoStatus.EMBEDDED not in statuses

    @pytest.mark.asyncio
    async def test_successful_run_publishes_event_and_marks_embedded(self, tmp_path):
        settings = _settings()
        model = AsyncMock()
        model.embed_image.return_value = [0.6, 0.8]
        storage = AsyncMock()
        storage.fetch_artifact.return_value = b"fake-bytes"
        publisher = AsyncMock()
        service = EmbeddingService(settings, model, storage, publisher)

        statuses: list[VideoStatus] = []

        async def fake_already_processed(video_id):
            return False

        async def fake_clear(*args, **kwargs):
            return None

        async def fake_mark_status(video_id, status, error_message=None):
            statuses.append(status)

        async def fake_write_rows(*args, **kwargs):
            return None

        async def fake_fetch_and_write(video_id, idx, artifact):
            local = tmp_path / f"{idx}.jpg"
            local.write_bytes(b"fake-bytes")
            return str(local)

        service._already_processed = fake_already_processed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]
        service._clear_existing_records = fake_clear  # type: ignore[assignment]
        service._write_embedding_rows = fake_write_rows  # type: ignore[assignment]
        service._fetch_and_write_artifact = fake_fetch_and_write  # type: ignore[assignment]

        video_id = uuid.uuid4()
        await service.process_detections(_detection_event(video_id))

        publisher.publish_embeddings_ready.assert_called_once()
        published_event = publisher.publish_embeddings_ready.call_args[0][0]
        assert published_event.video_id == str(video_id)
        assert len(published_event.embeddings) == 2  # one frame + one crop
        assert VideoStatus.EMBEDDED in statuses
        assert VideoStatus.FAILED not in statuses


class TestEmbeddingServiceVectorAssembly:
    @pytest.mark.asyncio
    async def test_embed_artifacts_calls_model_per_artifact_and_preserves_order(self, tmp_path):
        settings = _settings()
        model = AsyncMock()
        model.embed_image.side_effect = [[0.6, 0.8], [1.0, 0.0]]
        storage = AsyncMock()
        publisher = AsyncMock()
        service = EmbeddingService(settings, model, storage, publisher)

        async def fake_fetch_and_write(video_id, idx, artifact):
            local = tmp_path / f"{idx}.jpg"
            local.write_bytes(b"fake-bytes")
            return str(local)

        service._fetch_and_write_artifact = fake_fetch_and_write  # type: ignore[assignment]

        artifacts = [
            ArtifactRef(
                kind="frame",
                source_path="a.jpg",
                source_bucket="processed-frames",
                timestamp_ms=0,
                label=None,
            ),
            ArtifactRef(
                kind="crop",
                source_path="b.jpg",
                source_bucket="detection-crops",
                timestamp_ms=0,
                label="person",
            ),
        ]

        video_id = uuid.uuid4()
        persisted = await service._embed_artifacts(video_id, artifacts)

        assert len(persisted) == 2
        assert persisted[0].vector == [0.6, 0.8]
        assert persisted[0].artifact.kind == "frame"
        assert persisted[1].vector == [1.0, 0.0]
        assert persisted[1].artifact.kind == "crop"
        assert model.embed_image.call_count == 2


class TestEmbeddingStorageService:
    @pytest.mark.asyncio
    async def test_fetch_artifact_delegates_to_client(self):
        from services.embedding.services.storage import EmbeddingStorageService

        client = AsyncMock()
        client.get_object.return_value = b"image-bytes"
        settings = _settings()
        storage = EmbeddingStorageService(client, settings)

        data = await storage.fetch_artifact("processed-frames", "video/0.jpg")

        assert data == b"image-bytes"
        client.get_object.assert_called_once_with("processed-frames", "video/0.jpg")


class TestCLIPEmbedderNormalization:
    """CLIPEmbedder._normalize is a pure, torch-free function — testable
    without a loaded model, transformers, or a GPU."""

    def test_normalize_produces_unit_l2_norm(self):
        import numpy as np

        from services.embedding.services.clip_model import CLIPEmbedder

        vector = np.array([3.0, 4.0])
        normalized = CLIPEmbedder._normalize(vector)

        assert pytest.approx(float(np.linalg.norm(normalized)), rel=1e-6) == 1.0

    def test_normalize_handles_zero_vector_without_dividing_by_zero(self):
        import numpy as np

        from services.embedding.services.clip_model import CLIPEmbedder

        vector = np.array([0.0, 0.0])
        normalized = CLIPEmbedder._normalize(vector)

        assert normalized.tolist() == [0.0, 0.0]
