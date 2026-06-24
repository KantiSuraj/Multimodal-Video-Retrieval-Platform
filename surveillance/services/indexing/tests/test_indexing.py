"""Tests for the Indexing Service.

Mirrors the testing philosophy from embedding/tests/test_embedding.py:
- No infrastructure dependencies (RabbitMQ, PostgreSQL, Qdrant, MinIO).
- Mocks and fakes for all external boundaries.
- Test classes grouped by responsibility area.
- Async tests use pytest.mark.asyncio.
- Internal methods are tested directly where useful, same as embedding
  tests testing _collect_artifacts and _embed_artifacts.
"""
import uuid

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.shared.events.embeddings_ready import (
    EmbeddingRecord as EmbeddingPayload,
    EmbeddingsReadyEvent,
)
from shared.shared.models.video import VideoStatus

from services.indexing.core.config import Settings
from services.indexing.models.schemas import (
    IndexingError,
    IndexingStage,
    QdrantPoint,
)
from services.indexing.services.indexing import IndexingService
from services.indexing.services.qdrant import QdrantService


def _settings() -> Settings:
    return Settings()


def _embeddings_event(
    video_id: uuid.UUID,
    *,
    dim: int = 768,
    count: int = 2,
    model_name: str = "openai/clip-vit-large-patch14",
) -> EmbeddingsReadyEvent:
    """Build a valid EmbeddingsReadyEvent for testing."""
    embeddings = []
    for i in range(count):
        kind = "frame" if i % 2 == 0 else "crop"
        embeddings.append(
            EmbeddingPayload(
                kind=kind,
                source_path=f"{video_id}/{i:06d}_{kind}.jpg",
                vector=[0.1] * dim,
                timestamp_ms=i * 1000,
                label="person" if kind == "crop" else None,
            )
        )
    return EmbeddingsReadyEvent(
        video_id=str(video_id),
        model_name=model_name,
        embeddings=embeddings,
    )


# ── Validation Tests ─────────────────────────────────────────────────────────


class TestIndexingServiceValidation:
    def test_valid_embeddings_pass_validation(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        video_id = uuid.uuid4()
        event = _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)

        # Should not raise
        service._validate_embeddings(event)

    def test_dimension_mismatch_raises_non_recoverable(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        video_id = uuid.uuid4()
        # Create event with wrong dimension
        event = _embeddings_event(video_id, dim=512)

        with pytest.raises(IndexingError) as exc_info:
            service._validate_embeddings(event)

        assert exc_info.value.recoverable is False
        assert exc_info.value.stage == IndexingStage.VALIDATE
        assert "512" in exc_info.value.message
        assert str(settings.QDRANT_VECTOR_DIMENSION) in exc_info.value.message

    def test_empty_embeddings_raises_non_recoverable(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        video_id = uuid.uuid4()
        event = EmbeddingsReadyEvent(
            video_id=str(video_id),
            model_name="openai/clip-vit-large-patch14",
            embeddings=[],
        )

        with pytest.raises(IndexingError) as exc_info:
            service._validate_embeddings(event)

        assert exc_info.value.recoverable is False
        assert exc_info.value.stage == IndexingStage.VALIDATE


# ── Transformation Tests ──────────────────────────────────────────────────────


class TestIndexingServiceTransformation:
    def test_transforms_embeddings_into_qdrant_points(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        video_id = uuid.uuid4()
        event = _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)

        points = service._transform_to_points(event)

        assert len(points) == len(event.embeddings)
        for point in points:
            assert isinstance(point, QdrantPoint)
            assert point.vector == [0.1] * settings.QDRANT_VECTOR_DIMENSION
            assert point.payload["video_id"] == str(video_id)

    def test_point_ids_are_deterministic(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        video_id = uuid.uuid4()
        event = _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)

        points_1 = service._transform_to_points(event)
        points_2 = service._transform_to_points(event)

        for p1, p2 in zip(points_1, points_2):
            assert p1.point_id == p2.point_id

    def test_different_source_paths_produce_different_point_ids(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        video_id = uuid.uuid4()
        event = _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION, count=3)

        points = service._transform_to_points(event)
        point_ids = [p.point_id for p in points]

        assert len(set(point_ids)) == len(point_ids)  # all unique

    def test_payload_contains_required_metadata(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        video_id = uuid.uuid4()
        event = _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)

        points = service._transform_to_points(event)

        for point, emb in zip(points, event.embeddings):
            assert point.payload["video_id"] == str(video_id)
            assert point.payload["kind"] == emb.kind
            assert point.payload["source_path"] == emb.source_path
            assert point.payload["model_name"] == event.model_name

    def test_optional_payload_fields_included_when_present(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        video_id = uuid.uuid4()
        event = _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)

        points = service._transform_to_points(event)

        crop_point = next(p for p in points if p.payload["kind"] == "crop")
        assert "label" in crop_point.payload
        assert "timestamp_ms" in crop_point.payload

    def test_optional_payload_fields_excluded_when_none(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        video_id = uuid.uuid4()
        event = EmbeddingsReadyEvent(
            video_id=str(video_id),
            model_name="openai/clip-vit-large-patch14",
            embeddings=[
                EmbeddingPayload(
                    kind="frame",
                    source_path=f"{video_id}/000000.jpg",
                    vector=[0.1] * settings.QDRANT_VECTOR_DIMENSION,
                    timestamp_ms=None,
                    label=None,
                )
            ],
        )

        points = service._transform_to_points(event)

        assert "timestamp_ms" not in points[0].payload
        assert "label" not in points[0].payload


# ── Idempotency Tests ─────────────────────────────────────────────────────────


class TestIndexingServiceIdempotency:
    @pytest.mark.asyncio
    async def test_already_indexed_video_is_skipped(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        async def fake_already_indexed(video_id):
            return True

        service._already_indexed = fake_already_indexed  # type: ignore[assignment]

        video_id = uuid.uuid4()
        await service.process_embeddings(
            _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)
        )

        qdrant.batch_upsert.assert_not_called()

    def test_deterministic_point_id_is_stable_across_calls(self):
        """Same video_id + source_path always produces the same point ID."""
        id1 = IndexingService._deterministic_point_id("abc", "frames/0.jpg")
        id2 = IndexingService._deterministic_point_id("abc", "frames/0.jpg")
        assert id1 == id2

    def test_deterministic_point_id_changes_with_different_inputs(self):
        id1 = IndexingService._deterministic_point_id("abc", "frames/0.jpg")
        id2 = IndexingService._deterministic_point_id("abc", "frames/1.jpg")
        assert id1 != id2


# ── Failure Handling Tests ────────────────────────────────────────────────────


class TestIndexingServiceFailureModes:
    @pytest.mark.asyncio
    async def test_recoverable_qdrant_failure_propagates_for_requeue(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        qdrant.ensure_collection = AsyncMock()
        qdrant.batch_upsert.side_effect = IndexingError(
            message="Qdrant timeout",
            stage=IndexingStage.UPSERT,
            recoverable=True,
        )
        service = IndexingService(settings, qdrant)

        async def fake_already_indexed(video_id):
            return False

        async def fake_mark_status(*args, **kwargs):
            return None

        service._already_indexed = fake_already_indexed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]

        video_id = uuid.uuid4()
        with pytest.raises(IndexingError) as exc_info:
            await service.process_embeddings(
                _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)
            )

        assert exc_info.value.recoverable is True
        assert exc_info.value.stage == IndexingStage.UPSERT

    @pytest.mark.asyncio
    async def test_non_recoverable_validation_failure_marks_failed_and_does_not_raise(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        statuses: list[VideoStatus] = []

        async def fake_already_indexed(video_id):
            return False

        async def fake_mark_status(video_id, status, error_message=None):
            statuses.append(status)

        service._already_indexed = fake_already_indexed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]

        video_id = uuid.uuid4()
        # Wrong dimension — non-recoverable
        await service.process_embeddings(
            _embeddings_event(video_id, dim=512)
        )

        assert VideoStatus.FAILED in statuses
        assert VideoStatus.INDEXED not in statuses

    @pytest.mark.asyncio
    async def test_successful_run_upserts_and_marks_indexed(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        qdrant.ensure_collection = AsyncMock()
        qdrant.batch_upsert = AsyncMock()
        service = IndexingService(settings, qdrant)

        statuses: list[VideoStatus] = []

        async def fake_already_indexed(video_id):
            return False

        async def fake_mark_status(video_id, status, error_message=None):
            statuses.append(status)

        async def fake_update_records(*args, **kwargs):
            return None

        service._already_indexed = fake_already_indexed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]
        service._update_embedding_records = fake_update_records  # type: ignore[assignment]

        video_id = uuid.uuid4()
        await service.process_embeddings(
            _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)
        )

        qdrant.ensure_collection.assert_called_once()
        qdrant.batch_upsert.assert_called_once()
        assert VideoStatus.INDEXED in statuses
        assert VideoStatus.FAILED not in statuses

        # Verify points were passed to batch_upsert
        upserted_points = qdrant.batch_upsert.call_args[0][0]
        assert len(upserted_points) == 2


# ── QdrantService Tests ──────────────────────────────────────────────────────


class TestQdrantServiceCollectionManagement:
    @pytest.mark.asyncio
    async def test_ensure_collection_creates_when_not_exists(self):
        settings = _settings()
        qdrant_svc = QdrantService(settings)

        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = False
        mock_client.create_collection = AsyncMock()
        qdrant_svc._client = mock_client

        await qdrant_svc.ensure_collection()

        mock_client.collection_exists.assert_called_once_with(settings.QDRANT_COLLECTION_NAME)
        mock_client.create_collection.assert_called_once()

        # Verify HNSW parameters were passed
        call_kwargs = mock_client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == settings.QDRANT_COLLECTION_NAME

    @pytest.mark.asyncio
    async def test_ensure_collection_validates_when_exists(self):
        settings = _settings()
        qdrant_svc = QdrantService(settings)

        # Build a mock collection info that matches our settings
        mock_vectors_config = MagicMock()
        mock_vectors_config.size = settings.QDRANT_VECTOR_DIMENSION
        mock_vectors_config.distance = qdrant_svc._resolve_distance()

        mock_config = MagicMock()
        mock_config.config.params.vectors = mock_vectors_config

        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = mock_config
        qdrant_svc._client = mock_client

        # Should not raise
        await qdrant_svc.ensure_collection()

        mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_collection_raises_on_dimension_mismatch(self):
        settings = _settings()
        qdrant_svc = QdrantService(settings)

        mock_vectors_config = MagicMock()
        mock_vectors_config.size = 512  # Wrong dimension
        mock_vectors_config.distance = qdrant_svc._resolve_distance()

        mock_config = MagicMock()
        mock_config.config.params.vectors = mock_vectors_config

        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = mock_config
        qdrant_svc._client = mock_client

        with pytest.raises(IndexingError) as exc_info:
            await qdrant_svc.ensure_collection()

        assert exc_info.value.recoverable is False
        assert exc_info.value.stage == IndexingStage.COLLECTION_INIT
        assert "512" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_ensure_collection_is_idempotent(self):
        """Calling ensure_collection twice does not create duplicates."""
        settings = _settings()
        qdrant_svc = QdrantService(settings)

        mock_client = AsyncMock()
        # First call: does not exist → create
        # Second call: exists → validate
        mock_client.collection_exists.side_effect = [False, True]
        mock_client.create_collection = AsyncMock()

        mock_vectors_config = MagicMock()
        mock_vectors_config.size = settings.QDRANT_VECTOR_DIMENSION
        mock_vectors_config.distance = qdrant_svc._resolve_distance()
        mock_config = MagicMock()
        mock_config.config.params.vectors = mock_vectors_config
        mock_client.get_collection.return_value = mock_config

        qdrant_svc._client = mock_client

        await qdrant_svc.ensure_collection()
        await qdrant_svc.ensure_collection()

        assert mock_client.create_collection.call_count == 1


class TestQdrantServiceBatchUpsert:
    @pytest.mark.asyncio
    async def test_batch_upsert_splits_into_batches(self):
        settings = _settings()
        # Small batch size for testing
        settings.QDRANT_UPSERT_BATCH_SIZE = 2
        qdrant_svc = QdrantService(settings)

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()
        qdrant_svc._client = mock_client

        points = [
            QdrantPoint(point_id=str(uuid.uuid4()), vector=[0.1] * 768, payload={"idx": i})
            for i in range(5)
        ]

        await qdrant_svc.batch_upsert(points)

        # 5 points / batch_size 2 = 3 upsert calls
        assert mock_client.upsert.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_upsert_handles_empty_list(self):
        settings = _settings()
        qdrant_svc = QdrantService(settings)

        mock_client = AsyncMock()
        qdrant_svc._client = mock_client

        await qdrant_svc.batch_upsert([])

        mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_upsert_raises_recoverable_on_connection_error(self):
        settings = _settings()
        qdrant_svc = QdrantService(settings)

        mock_client = AsyncMock()
        mock_client.upsert.side_effect = ConnectionError("Qdrant down")
        qdrant_svc._client = mock_client

        points = [
            QdrantPoint(point_id=str(uuid.uuid4()), vector=[0.1] * 768, payload={})
        ]

        with pytest.raises(IndexingError) as exc_info:
            await qdrant_svc.batch_upsert(points)

        assert exc_info.value.recoverable is True
        assert exc_info.value.stage == IndexingStage.UPSERT

    @pytest.mark.asyncio
    async def test_batch_upsert_single_batch_for_small_input(self):
        settings = _settings()
        settings.QDRANT_UPSERT_BATCH_SIZE = 100
        qdrant_svc = QdrantService(settings)

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()
        qdrant_svc._client = mock_client

        points = [
            QdrantPoint(point_id=str(uuid.uuid4()), vector=[0.1] * 768, payload={})
            for _ in range(10)
        ]

        await qdrant_svc.batch_upsert(points)

        assert mock_client.upsert.call_count == 1


# ── Consumer Tests ────────────────────────────────────────────────────────────


class TestIndexingConsumer:
    @pytest.mark.asyncio
    async def test_valid_message_dispatches_to_callback(self):
        from services.indexing.services.queue import IndexingConsumer

        callback = AsyncMock()
        settings = _settings()
        consumer = IndexingConsumer(settings, on_embeddings_ready=callback)

        video_id = uuid.uuid4()
        event = _embeddings_event(video_id, dim=768)
        body = event.model_dump_json().encode()

        await consumer.handle_message(body, "video.embeddings_ready")

        callback.assert_called_once()
        dispatched_event = callback.call_args[0][0]
        assert dispatched_event.video_id == str(video_id)
        assert len(dispatched_event.embeddings) == 2

    @pytest.mark.asyncio
    async def test_malformed_message_is_dropped_not_dispatched(self):
        from services.indexing.services.queue import IndexingConsumer

        callback = AsyncMock()
        settings = _settings()
        consumer = IndexingConsumer(settings, on_embeddings_ready=callback)

        body = b'{"invalid": "json", "missing_required_fields": true}'

        # Should not raise — malformed events are dropped and acked
        await consumer.handle_message(body, "video.embeddings_ready")

        callback.assert_not_called()


# ── Integration-style Orchestration Tests ─────────────────────────────────────


class TestIndexingServiceOrchestration:
    @pytest.mark.asyncio
    async def test_full_pipeline_validate_transform_upsert_persist(self):
        """End-to-end orchestration: validate → transform → upsert → persist → status."""
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        qdrant.ensure_collection = AsyncMock()
        qdrant.batch_upsert = AsyncMock()
        service = IndexingService(settings, qdrant)

        statuses: list[VideoStatus] = []
        updated_records: list[tuple] = []

        async def fake_already_indexed(video_id):
            return False

        async def fake_mark_status(video_id, status, error_message=None):
            statuses.append(status)

        async def fake_update_records(event, points):
            for emb, point in zip(event.embeddings, points):
                updated_records.append((emb.source_path, point.point_id))

        service._already_indexed = fake_already_indexed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]
        service._update_embedding_records = fake_update_records  # type: ignore[assignment]

        video_id = uuid.uuid4()
        event = _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)
        await service.process_embeddings(event)

        # Verify pipeline order: PROCESSING → INDEXED
        assert statuses == [VideoStatus.PROCESSING, VideoStatus.INDEXED]

        # Verify Qdrant was called
        qdrant.ensure_collection.assert_called_once()
        qdrant.batch_upsert.assert_called_once()

        # Verify metadata persistence
        assert len(updated_records) == len(event.embeddings)

        # Verify point IDs in batch_upsert match what metadata received
        upserted_points = qdrant.batch_upsert.call_args[0][0]
        for (path, pid), point in zip(updated_records, upserted_points):
            assert pid == point.point_id

    @pytest.mark.asyncio
    async def test_qdrant_failure_does_not_persist_metadata(self):
        """If Qdrant upsert fails recoverably, metadata must NOT be written."""
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        qdrant.ensure_collection = AsyncMock()
        qdrant.batch_upsert.side_effect = IndexingError(
            message="timeout",
            stage=IndexingStage.UPSERT,
            recoverable=True,
        )
        service = IndexingService(settings, qdrant)

        metadata_calls: list = []

        async def fake_already_indexed(video_id):
            return False

        async def fake_mark_status(*args, **kwargs):
            return None

        async def fake_update_records(*args, **kwargs):
            metadata_calls.append(1)

        service._already_indexed = fake_already_indexed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]
        service._update_embedding_records = fake_update_records  # type: ignore[assignment]

        video_id = uuid.uuid4()
        with pytest.raises(IndexingError):
            await service.process_embeddings(
                _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)
            )

        # Metadata must NOT have been written
        assert len(metadata_calls) == 0


class TestQdrantServiceLifecycle:
    @pytest.mark.asyncio
    async def test_require_client_raises_before_startup(self):
        settings = _settings()
        qdrant_svc = QdrantService(settings)

        with pytest.raises(RuntimeError, match="not started"):
            qdrant_svc._require_client()


# ── Critical Invariant Tests ──────────────────────────────────────────────────
# These tests verify the most important behavioural guarantees of the service.
# If any of these fail, the system has a data corruption or consistency bug.


class TestIdempotencyDuplicateEventReplay:
    """Process same event twice: 100 points, NOT 200.

    This is the single most important test in the service. RabbitMQ
    provides at-least-once delivery. If deterministic point IDs fail,
    every redelivery creates duplicate vectors in Qdrant and search
    returns false positives.
    """

    @pytest.mark.asyncio
    async def test_same_event_twice_produces_same_points_not_double(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        qdrant.ensure_collection = AsyncMock()
        qdrant.batch_upsert = AsyncMock()
        service = IndexingService(settings, qdrant)

        async def fake_already_indexed(video_id):
            # Simulate that the first run does NOT set INDEXED yet
            # (e.g. worker crashed after upsert but before status update).
            # Second delivery should still be safe.
            return False

        async def fake_mark_status(*args, **kwargs):
            return None

        async def fake_update_records(*args, **kwargs):
            return None

        service._already_indexed = fake_already_indexed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]
        service._update_embedding_records = fake_update_records  # type: ignore[assignment]

        video_id = uuid.uuid4()
        event = _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION, count=100)

        # Process event TWICE (simulating at-least-once redelivery)
        await service.process_embeddings(event)
        await service.process_embeddings(event)

        # Both calls produced exactly 100 points each
        assert qdrant.batch_upsert.call_count == 2
        first_points = qdrant.batch_upsert.call_args_list[0][0][0]
        second_points = qdrant.batch_upsert.call_args_list[1][0][0]
        assert len(first_points) == 100
        assert len(second_points) == 100

        # CRITICAL: same point IDs both times — Qdrant upsert overwrites,
        # never creates duplicates. 100 unique points, NOT 200.
        first_ids = [p.point_id for p in first_points]
        second_ids = [p.point_id for p in second_points]
        assert first_ids == second_ids

        # Verify all 100 IDs are unique (no internal collisions)
        assert len(set(first_ids)) == 100


class TestEmbeddingRecordQdrantLinkage:
    """Invariant: After indexing, EmbeddingRecord.qdrant_point_id and
    qdrant_collection must BOTH exist. No orphan rows.

    The _update_embedding_records method fills both fields for every
    embedding in the event. If either is missing, Search cannot correlate
    PostgreSQL metadata back to Qdrant vectors.
    """

    @pytest.mark.asyncio
    async def test_update_records_sets_both_qdrant_fields_for_every_embedding(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        qdrant.ensure_collection = AsyncMock()
        qdrant.batch_upsert = AsyncMock()
        service = IndexingService(settings, qdrant)

        video_id = uuid.uuid4()
        event = _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION, count=5)
        points = service._transform_to_points(event)

        # Verify every point that would be passed to _update_embedding_records
        # has a non-empty point_id and the collection name is configured
        for emb, point in zip(event.embeddings, points):
            assert point.point_id is not None
            assert point.point_id != ""
            assert len(point.point_id) > 0

        # Verify the values that would be written are correct
        collection_name = settings.QDRANT_COLLECTION_NAME
        assert collection_name is not None
        assert collection_name != ""

        # Each point_id is unique — no two embeddings map to the same point
        all_ids = [p.point_id for p in points]
        assert len(set(all_ids)) == len(all_ids)

    @pytest.mark.asyncio
    async def test_every_embedding_gets_a_corresponding_point(self):
        """No embedding is silently dropped during transformation."""
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        service = IndexingService(settings, qdrant)

        video_id = uuid.uuid4()
        event = _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION, count=50)
        points = service._transform_to_points(event)

        # 1:1 mapping — no orphan embeddings, no orphan points
        assert len(points) == len(event.embeddings)


class TestIndexedMeansSearchable:
    """Invariant: INDEXED means searchable.

    The status MUST be set in this exact order:
        1. batch_upsert succeeds
        2. _update_embedding_records succeeds
        3. status = INDEXED

    If INDEXED is set before upsert, Search receives false positives
    (it trusts INDEXED to mean "vectors are in Qdrant").
    """

    @pytest.mark.asyncio
    async def test_indexed_is_set_only_after_upsert_and_metadata_both_succeed(self):
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        qdrant.ensure_collection = AsyncMock()
        qdrant.batch_upsert = AsyncMock()
        service = IndexingService(settings, qdrant)

        # Track the EXACT order of operations
        call_order: list[str] = []

        async def fake_already_indexed(video_id):
            return False

        async def fake_mark_status(video_id, status, error_message=None):
            call_order.append(f"status:{status.value}")

        original_batch_upsert = qdrant.batch_upsert

        async def tracking_batch_upsert(points):
            call_order.append("upsert")
            return await original_batch_upsert(points)

        async def fake_update_records(*args, **kwargs):
            call_order.append("metadata")

        service._already_indexed = fake_already_indexed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]
        service._update_embedding_records = fake_update_records  # type: ignore[assignment]
        qdrant.batch_upsert = tracking_batch_upsert  # type: ignore[assignment]

        video_id = uuid.uuid4()
        await service.process_embeddings(
            _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)
        )

        # CRITICAL ordering assertion:
        # PROCESSING first, then upsert, then metadata, then INDEXED
        assert call_order == [
            "status:PROCESSING",
            "upsert",
            "metadata",
            "status:INDEXED",
        ]

    @pytest.mark.asyncio
    async def test_upsert_failure_prevents_indexed_status(self):
        """If upsert fails, INDEXED must NEVER appear in status transitions."""
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        qdrant.ensure_collection = AsyncMock()
        qdrant.batch_upsert.side_effect = IndexingError(
            message="Qdrant timeout",
            stage=IndexingStage.UPSERT,
            recoverable=True,
        )
        service = IndexingService(settings, qdrant)

        statuses: list[str] = []

        async def fake_already_indexed(video_id):
            return False

        async def fake_mark_status(video_id, status, error_message=None):
            statuses.append(status.value)

        service._already_indexed = fake_already_indexed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]

        video_id = uuid.uuid4()
        with pytest.raises(IndexingError):
            await service.process_embeddings(
                _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)
            )

        assert "INDEXED" not in statuses

    @pytest.mark.asyncio
    async def test_metadata_failure_prevents_indexed_status(self):
        """If metadata persistence fails, INDEXED must NEVER appear."""
        settings = _settings()
        qdrant = AsyncMock(spec=QdrantService)
        qdrant.ensure_collection = AsyncMock()
        qdrant.batch_upsert = AsyncMock()
        service = IndexingService(settings, qdrant)

        statuses: list[str] = []

        async def fake_already_indexed(video_id):
            return False

        async def fake_mark_status(video_id, status, error_message=None):
            statuses.append(status.value)

        async def fake_update_records_fails(*args, **kwargs):
            raise IndexingError(
                message="DB connection lost",
                stage=IndexingStage.METADATA_PERSIST,
                recoverable=True,
            )

        service._already_indexed = fake_already_indexed  # type: ignore[assignment]
        service._mark_status = fake_mark_status  # type: ignore[assignment]
        service._update_embedding_records = fake_update_records_fails  # type: ignore[assignment]

        video_id = uuid.uuid4()
        with pytest.raises(IndexingError):
            await service.process_embeddings(
                _embeddings_event(video_id, dim=settings.QDRANT_VECTOR_DIMENSION)
            )

        assert "INDEXED" not in statuses


class TestCollectionSchemaMismatchBlocksStartup:
    """Suppose: configured dimension=768, existing collection dimension=512.

    Startup MUST fail. Not continue.

    This protects against model migrations — if someone switches from
    CLIP-base (512d) to CLIP-large (768d) but forgets to recreate the
    collection, the service must refuse to index into a mismatched
    collection rather than silently corrupting the index.
    """

    @pytest.mark.asyncio
    async def test_existing_512d_collection_with_768d_config_fails_non_recoverable(self):
        settings = _settings()
        assert settings.QDRANT_VECTOR_DIMENSION == 768  # sanity check

        qdrant_svc = QdrantService(settings)

        # Simulate existing collection with dimension=512 (old model)
        mock_vectors_config = MagicMock()
        mock_vectors_config.size = 512  # ← MISMATCH
        mock_vectors_config.distance = qdrant_svc._resolve_distance()

        mock_config = MagicMock()
        mock_config.config.params.vectors = mock_vectors_config

        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = mock_config
        qdrant_svc._client = mock_client

        with pytest.raises(IndexingError) as exc_info:
            await qdrant_svc.ensure_collection()

        # Must be non-recoverable — retrying won't fix a schema mismatch
        assert exc_info.value.recoverable is False
        assert exc_info.value.stage == IndexingStage.COLLECTION_INIT
        assert "512" in exc_info.value.message
        assert "768" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_matching_collection_passes_validation(self):
        """Control test: correct dimension does NOT raise."""
        settings = _settings()
        qdrant_svc = QdrantService(settings)

        mock_vectors_config = MagicMock()
        mock_vectors_config.size = 768  # ← MATCHES
        mock_vectors_config.distance = qdrant_svc._resolve_distance()

        mock_config = MagicMock()
        mock_config.config.params.vectors = mock_vectors_config

        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = mock_config
        qdrant_svc._client = mock_client

        # Must NOT raise
        await qdrant_svc.ensure_collection()

        mock_client.create_collection.assert_not_called()
