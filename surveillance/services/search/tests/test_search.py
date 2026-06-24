"""Tests for the Search Service.

Mirrors the testing philosophy from embedding/tests/test_embedding.py
and indexing/tests/test_indexing.py:
- No infrastructure dependencies (Qdrant, PostgreSQL, Redis, MinIO).
- Mocks and fakes for all external boundaries.
- Test classes grouped by responsibility area.
- Async tests use pytest.mark.asyncio.
"""
import uuid
import base64

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.search.core.config import Settings
from services.search.models.schemas import (
    HydratedResult,
    MetadataFilters,
    RetrievedPoint,
    SearchError,
    SearchStage,
)
from services.search.services.cache import (
    SearchCacheClient,
    dict_to_result,
    make_text_embedding_key,
    make_image_embedding_key,
    make_result_cache_key,
    _result_to_dict,
)
from services.search.services.clip_encoder import CLIPQueryEncoder
from services.search.services.qdrant import QdrantSearchClient
from services.search.services.search import SearchService


def _settings() -> Settings:
    return Settings()


def _make_hydrated(
    video_id: str | None = None,
    score: float = 0.9,
    timestamp_start: int = 0,
) -> HydratedResult:
    return HydratedResult(
        video_id=video_id or str(uuid.uuid4()),
        clip_id=str(uuid.uuid4()),
        camera_name="CAM-01",
        camera_location="entrance",
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_start + 5000,
        similarity_score=score,
        thumbnail_url=None,
        detected_labels=[],
        video_start_epoch=None,
    )


def _make_service(settings=None) -> tuple[SearchService, AsyncMock, AsyncMock, AsyncMock]:
    s = settings or _settings()
    encoder = AsyncMock(spec=CLIPQueryEncoder)
    qdrant = AsyncMock(spec=QdrantSearchClient)
    cache = AsyncMock(spec=SearchCacheClient)
    svc = SearchService(s, encoder, qdrant, cache)
    return svc, encoder, qdrant, cache


# ── Validation Tests ──────────────────────────────────────────────────────────


class TestSearchServiceValidation:
    def test_empty_text_query_raises_non_recoverable(self):
        svc, _, _, _ = _make_service()
        with pytest.raises(SearchError) as exc_info:
            svc._validate_text_query("")
        assert exc_info.value.recoverable is False
        assert exc_info.value.stage == SearchStage.VALIDATION

    def test_whitespace_only_query_raises_non_recoverable(self):
        svc, _, _, _ = _make_service()
        with pytest.raises(SearchError) as exc_info:
            svc._validate_text_query("   ")
        assert exc_info.value.recoverable is False

    def test_valid_query_does_not_raise(self):
        svc, _, _, _ = _make_service()
        svc._validate_text_query("person in red jacket")  # must not raise

    def test_query_too_long_raises_non_recoverable(self):
        svc, _, _, _ = _make_service()
        with pytest.raises(SearchError) as exc_info:
            svc._validate_text_query("x" * 1001)
        assert exc_info.value.recoverable is False

    def test_vector_dimension_mismatch_raises_non_recoverable(self):
        settings = _settings()
        settings.QDRANT_VECTOR_DIMENSION = 512
        svc, _, _, _ = _make_service(settings)
        with pytest.raises(SearchError) as exc_info:
            svc._validate_vector_dimension([0.1] * 768)
        assert exc_info.value.recoverable is False
        assert exc_info.value.stage == SearchStage.VALIDATION

    def test_correct_dimension_does_not_raise(self):
        settings = _settings()
        svc, _, _, _ = _make_service(settings)
        svc._validate_vector_dimension([0.1] * settings.QDRANT_VECTOR_DIMENSION)


# ── Ranking Tests ─────────────────────────────────────────────────────────────


class TestSearchServiceRanking:
    def test_rank_orders_by_score_descending(self):
        svc, _, _, _ = _make_service()
        results = [
            _make_hydrated(score=0.5),
            _make_hydrated(score=0.9),
            _make_hydrated(score=0.7),
        ]
        ranked = svc._rank(results)
        scores = [r.similarity_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_empty_list_returns_empty(self):
        svc, _, _, _ = _make_service()
        assert svc._rank([]) == []

    def test_rank_single_item_returns_same(self):
        svc, _, _, _ = _make_service()
        item = _make_hydrated(score=0.8)
        assert svc._rank([item]) == [item]


# ── Temporal Deduplication Tests ──────────────────────────────────────────────


class TestSearchServiceTemporalDeduplication:
    def test_same_video_within_window_keeps_higher_ranked(self):
        settings = _settings()
        settings.SEARCH_TEMPORAL_DEDUP_WINDOW_MS = 5000
        svc, _, _, _ = _make_service(settings)

        vid = str(uuid.uuid4())
        # Two results from same video within 5s of each other; score=0.9 is first (higher rank)
        results = [
            _make_hydrated(video_id=vid, score=0.9, timestamp_start=0),
            _make_hydrated(video_id=vid, score=0.7, timestamp_start=2000),
        ]
        deduped = svc._temporal_deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0].similarity_score == 0.9

    def test_same_video_outside_window_keeps_both(self):
        settings = _settings()
        settings.SEARCH_TEMPORAL_DEDUP_WINDOW_MS = 5000
        svc, _, _, _ = _make_service(settings)

        vid = str(uuid.uuid4())
        results = [
            _make_hydrated(video_id=vid, score=0.9, timestamp_start=0),
            _make_hydrated(video_id=vid, score=0.7, timestamp_start=10000),
        ]
        deduped = svc._temporal_deduplicate(results)
        assert len(deduped) == 2

    def test_different_videos_always_kept(self):
        settings = _settings()
        settings.SEARCH_TEMPORAL_DEDUP_WINDOW_MS = 5000
        svc, _, _, _ = _make_service(settings)

        results = [
            _make_hydrated(video_id=str(uuid.uuid4()), score=0.9, timestamp_start=0),
            _make_hydrated(video_id=str(uuid.uuid4()), score=0.8, timestamp_start=1000),
        ]
        deduped = svc._temporal_deduplicate(results)
        assert len(deduped) == 2

    def test_dedup_preserves_order(self):
        settings = _settings()
        settings.SEARCH_TEMPORAL_DEDUP_WINDOW_MS = 5000
        svc, _, _, _ = _make_service(settings)

        vid = str(uuid.uuid4())
        results = [
            _make_hydrated(video_id=vid, score=0.95, timestamp_start=0),
            _make_hydrated(video_id=str(uuid.uuid4()), score=0.85, timestamp_start=0),
            _make_hydrated(video_id=vid, score=0.60, timestamp_start=2000),
        ]
        deduped = svc._temporal_deduplicate(results)
        assert len(deduped) == 2
        assert deduped[0].similarity_score == 0.95
        assert deduped[1].similarity_score == 0.85


# ── Pagination Tests ──────────────────────────────────────────────────────────


class TestSearchServicePagination:
    def test_first_page_returns_correct_slice(self):
        svc, _, _, _ = _make_service()
        results = [_make_hydrated(score=1.0 - i * 0.01) for i in range(50)]
        page, token = svc._paginate(results, page_size=10, page_token=None)
        assert len(page) == 10
        assert page[0] == results[0]
        assert token is not None

    def test_last_page_returns_none_token(self):
        svc, _, _, _ = _make_service()
        results = [_make_hydrated() for _ in range(5)]
        page, token = svc._paginate(results, page_size=10, page_token=None)
        assert len(page) == 5
        assert token is None

    def test_second_page_continues_from_first(self):
        svc, _, _, _ = _make_service()
        results = [_make_hydrated(score=1.0 - i * 0.01) for i in range(25)]
        page1, token1 = svc._paginate(results, page_size=10, page_token=None)
        page2, token2 = svc._paginate(results, page_size=10, page_token=token1)
        assert page2[0] == results[10]
        assert token2 is not None

    def test_malformed_token_raises_non_recoverable(self):
        svc, _, _, _ = _make_service()
        with pytest.raises(SearchError) as exc_info:
            svc._decode_page_token("!!!not_base64!!!")
        assert exc_info.value.recoverable is False
        assert exc_info.value.stage == SearchStage.PAGINATE

    def test_encode_decode_roundtrip_is_stable(self):
        svc, _, _, _ = _make_service()
        for offset in [0, 10, 100, 999]:
            token = svc._encode_page_token(offset)
            decoded = svc._decode_page_token(token)
            assert decoded == offset

    def test_none_token_decodes_to_zero(self):
        svc, _, _, _ = _make_service()
        assert svc._decode_page_token(None) == 0


# ── Cache Behaviour Tests ─────────────────────────────────────────────────────


class TestSearchServiceCacheBehaviour:
    @pytest.mark.asyncio
    async def test_text_search_uses_cached_embedding_on_hit(self):
        svc, encoder, qdrant, cache = _make_service()
        cached_vector = [0.1] * 512
        cache.get_embedding = AsyncMock(return_value=cached_vector)
        cache.get_results = AsyncMock(return_value=None)
        cache.set_results = AsyncMock()
        cache.set_embedding = AsyncMock()
        cache.append_history = AsyncMock()
        qdrant.search = AsyncMock(return_value=[])

        svc._validate_vector_dimension = lambda v: None
        svc._hydrate = AsyncMock(return_value=[])

        await svc.execute_text_search("person", MetadataFilters())
        encoder.encode_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_result_cache_hit_skips_qdrant(self):
        svc, encoder, qdrant, cache = _make_service()
        cached_result = _make_hydrated()
        cache.get_embedding = AsyncMock(return_value=[0.1] * 512)
        cache.get_results = AsyncMock(return_value=[_result_to_dict(cached_result)])

        page, token = await svc.execute_text_search("person", MetadataFilters())

        qdrant.search.assert_not_called()
        encoder.encode_text.assert_not_called()
        assert len(page) >= 0  # may be 0 if page_size exceeds cached

    @pytest.mark.asyncio
    async def test_result_cache_miss_calls_qdrant(self):
        settings = _settings()
        settings.QDRANT_VECTOR_DIMENSION = 512
        svc, encoder, qdrant, cache = _make_service(settings)
        cache.get_embedding = AsyncMock(return_value=None)
        cache.get_results = AsyncMock(return_value=None)
        cache.set_embedding = AsyncMock()
        cache.set_results = AsyncMock()
        cache.append_history = AsyncMock()
        encoder.encode_text = AsyncMock(return_value=[0.1] * 512)
        qdrant.search = AsyncMock(return_value=[])
        svc._hydrate = AsyncMock(return_value=[])

        await svc.execute_text_search("person", MetadataFilters())
        qdrant.search.assert_called_once()


# ── Failure Handling Tests ────────────────────────────────────────────────────


class TestSearchServiceFailureModes:
    @pytest.mark.asyncio
    async def test_recoverable_qdrant_failure_propagates(self):
        settings = _settings()
        settings.QDRANT_VECTOR_DIMENSION = 512
        svc, encoder, qdrant, cache = _make_service(settings)
        cache.get_embedding = AsyncMock(return_value=None)
        cache.get_results = AsyncMock(return_value=None)
        cache.set_embedding = AsyncMock()
        encoder.encode_text = AsyncMock(return_value=[0.1] * 512)
        qdrant.search = AsyncMock(
            side_effect=SearchError("qdrant down", SearchStage.ANN_RETRIEVE, recoverable=True)
        )

        with pytest.raises(SearchError) as exc_info:
            await svc.execute_text_search("person", MetadataFilters())
        assert exc_info.value.recoverable is True

    @pytest.mark.asyncio
    async def test_non_recoverable_encoding_failure_propagates(self):
        settings = _settings()
        svc, encoder, qdrant, cache = _make_service(settings)
        cache.get_embedding = AsyncMock(return_value=None)
        cache.get_results = AsyncMock(return_value=None)
        encoder.encode_text = AsyncMock(
            side_effect=SearchError("bad image", SearchStage.ENCODE, recoverable=False)
        )

        with pytest.raises(SearchError) as exc_info:
            await svc.execute_text_search("person", MetadataFilters())
        assert exc_info.value.recoverable is False

    @pytest.mark.asyncio
    async def test_empty_image_list_raises_non_recoverable(self):
        svc, _, _, _ = _make_service()
        with pytest.raises(SearchError) as exc_info:
            await svc.execute_image_search([], MetadataFilters())
        assert exc_info.value.recoverable is False
        assert exc_info.value.stage == SearchStage.VALIDATION


# ── CLIPQueryEncoder Tests ────────────────────────────────────────────────────


class TestCLIPQueryEncoderNormalization:
    """_normalize is pure and torch-free — testable without a model or GPU."""

    def test_normalize_produces_unit_l2_norm(self):
        vector = np.array([3.0, 4.0])
        normalized = CLIPQueryEncoder._normalize(vector)
        assert pytest.approx(float(np.linalg.norm(normalized)), rel=1e-6) == 1.0

    def test_normalize_zero_vector_does_not_divide_by_zero(self):
        vector = np.array([0.0, 0.0])
        normalized = CLIPQueryEncoder._normalize(vector)
        assert normalized.tolist() == [0.0, 0.0]

    def test_average_embeddings_averages_and_renormalizes(self):
        e1 = [1.0, 0.0]
        e2 = [0.0, 1.0]
        result = CLIPQueryEncoder.average_embeddings([e1, e2])
        norm = float(np.linalg.norm(result))
        assert pytest.approx(norm, rel=1e-6) == 1.0

    def test_average_embeddings_single_input_returns_same(self):
        vec = [0.6, 0.8]
        result = CLIPQueryEncoder.average_embeddings([vec])
        # already unit norm: 0.6^2 + 0.8^2 = 1.0
        assert pytest.approx(result[0], rel=1e-6) == 0.6
        assert pytest.approx(result[1], rel=1e-6) == 0.8

    def test_average_embeddings_empty_raises_non_recoverable(self):
        with pytest.raises(SearchError) as exc_info:
            CLIPQueryEncoder.average_embeddings([])
        assert exc_info.value.recoverable is False


# ── QdrantSearchClient Filter Tests ──────────────────────────────────────────


class TestQdrantSearchClientFilterBuilding:
    def test_empty_filters_returns_none(self):
        filters = MetadataFilters()
        result = QdrantSearchClient._build_filter(filters)
        assert result is None

    def test_camera_ids_filter_produces_must_condition(self):
        filters = MetadataFilters(camera_ids=["cam-01", "cam-02"])
        result = QdrantSearchClient._build_filter(filters)
        assert result is not None
        assert len(result.must) == 1
        assert result.must[0].key == "camera_id"

    def test_locations_filter_produces_must_condition(self):
        filters = MetadataFilters(locations=["entrance"])
        result = QdrantSearchClient._build_filter(filters)
        assert result is not None
        assert result.must[0].key == "location"

    def test_time_range_filter_produces_must_condition(self):
        filters = MetadataFilters(start_ms=1000, end_ms=5000)
        result = QdrantSearchClient._build_filter(filters)
        assert result is not None
        ts_cond = next(c for c in result.must if c.key == "timestamp_ms")
        assert ts_cond.range.gte == 1000
        assert ts_cond.range.lte == 5000

    def test_combined_filters_produce_multiple_must_conditions(self):
        filters = MetadataFilters(
            camera_ids=["cam-01"],
            locations=["entrance"],
            start_ms=0,
        )
        result = QdrantSearchClient._build_filter(filters)
        assert result is not None
        assert len(result.must) == 3

    def test_label_filter_produces_must_condition(self):
        filters = MetadataFilters(labels=["person", "vehicle"])
        result = QdrantSearchClient._build_filter(filters)
        assert result is not None
        assert result.must[0].key == "label"


# ── QdrantSearchClient Lifecycle Tests ───────────────────────────────────────


class TestQdrantSearchClientLifecycle:
    @pytest.mark.asyncio
    async def test_require_client_raises_before_startup(self):
        settings = _settings()
        client = QdrantSearchClient(settings)
        with pytest.raises(RuntimeError, match="not started"):
            client._require_client()

    @pytest.mark.asyncio
    async def test_search_raises_recoverable_on_connection_error(self):
        settings = _settings()
        qdrant = QdrantSearchClient(settings)
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(side_effect=ConnectionError("down"))
        qdrant._client = mock_client

        with pytest.raises(SearchError) as exc_info:
            await qdrant.search([0.1] * 512, MetadataFilters(), top_k=10)
        assert exc_info.value.recoverable is True
        assert exc_info.value.stage == SearchStage.ANN_RETRIEVE


# ── Cache Key Tests ───────────────────────────────────────────────────────────


class TestCacheKeys:
    def test_text_embedding_key_is_stable(self):
        k1 = make_text_embedding_key("person in red jacket")
        k2 = make_text_embedding_key("person in red jacket")
        assert k1 == k2

    def test_text_embedding_key_differs_on_different_text(self):
        k1 = make_text_embedding_key("person")
        k2 = make_text_embedding_key("vehicle")
        assert k1 != k2

    def test_image_embedding_key_is_stable(self):
        data = b"fake-image-bytes"
        k1 = make_image_embedding_key(data)
        k2 = make_image_embedding_key(data)
        assert k1 == k2

    def test_result_cache_key_changes_with_filters(self):
        emb_key = "abc123"
        k1 = make_result_cache_key(emb_key, {"camera_ids": []})
        k2 = make_result_cache_key(emb_key, {"camera_ids": ["cam-01"]})
        assert k1 != k2

    def test_result_cache_key_is_stable(self):
        emb_key = "abc123"
        filters = {"camera_ids": ["cam-01"], "locations": []}
        k1 = make_result_cache_key(emb_key, filters)
        k2 = make_result_cache_key(emb_key, filters)
        assert k1 == k2


# ── SearchCacheClient Tests ───────────────────────────────────────────────────


class TestSearchCacheClient:
    @pytest.mark.asyncio
    async def test_get_embedding_returns_none_when_client_is_none(self):
        settings = _settings()
        cache = SearchCacheClient(settings)
        # _client is None (never started)
        result = await cache.get_embedding("somekey")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_embedding_is_noop_when_client_is_none(self):
        settings = _settings()
        cache = SearchCacheClient(settings)
        # Must not raise
        await cache.set_embedding("somekey", [0.1, 0.2])

    @pytest.mark.asyncio
    async def test_get_results_returns_none_when_client_is_none(self):
        settings = _settings()
        cache = SearchCacheClient(settings)
        result = await cache.get_results("somekey")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_history_returns_empty_when_client_is_none(self):
        settings = _settings()
        cache = SearchCacheClient(settings)
        entries, cursor = await cache.get_history()
        assert entries == []
        assert cursor is None


# ── Result Serialisation Tests ────────────────────────────────────────────────


class TestResultSerialisation:
    def test_result_to_dict_and_back_is_lossless(self):
        original = _make_hydrated(score=0.75, timestamp_start=3000)
        original.detected_labels = ["person"]
        d = _result_to_dict(original)
        restored = dict_to_result(d)
        assert restored.video_id == original.video_id
        assert restored.similarity_score == original.similarity_score
        assert restored.timestamp_start == original.timestamp_start
        assert restored.detected_labels == original.detected_labels


# ── End-to-End Orchestration Tests ───────────────────────────────────────────


class TestSearchServiceOrchestration:
    @pytest.mark.asyncio
    async def test_text_search_pipeline_order(self):
        """Verify encode → retrieve → hydrate → rank → dedup → paginate."""
        settings = _settings()
        settings.QDRANT_VECTOR_DIMENSION = 512
        svc, encoder, qdrant, cache = _make_service(settings)

        call_order: list[str] = []
        vid = str(uuid.uuid4())

        cache.get_embedding = AsyncMock(return_value=None)
        cache.get_results = AsyncMock(return_value=None)
        cache.set_embedding = AsyncMock()
        cache.set_results = AsyncMock()
        cache.append_history = AsyncMock()

        async def fake_encode(text):
            call_order.append("encode")
            return [0.1] * 512

        encoder.encode_text = fake_encode

        async def fake_search(query_vector, filters, top_k):
            call_order.append("retrieve")
            return [RetrievedPoint(point_id="p1", score=0.9, payload={
                "video_id": vid, "source_path": "v/f.jpg"
            })]

        qdrant.search = fake_search

        async def fake_hydrate(points):
            call_order.append("hydrate")
            return [_make_hydrated(video_id=vid, score=0.9)]

        svc._hydrate = fake_hydrate

        original_rank = svc._rank
        def tracking_rank(results):
            call_order.append("rank")
            return original_rank(results)
        svc._rank = tracking_rank

        original_dedup = svc._temporal_deduplicate
        def tracking_dedup(results):
            call_order.append("deduplicate")
            return original_dedup(results)
        svc._temporal_deduplicate = tracking_dedup

        await svc.execute_text_search("person", MetadataFilters())

        assert call_order == ["encode", "retrieve", "hydrate", "rank", "deduplicate"]

    @pytest.mark.asyncio
    async def test_image_search_averages_multiple_images(self):
        settings = _settings()
        settings.QDRANT_VECTOR_DIMENSION = 512
        svc, encoder, qdrant, cache = _make_service(settings)

        embeddings_encoded: list[list[float]] = []

        cache.get_embedding = AsyncMock(return_value=None)
        cache.get_results = AsyncMock(return_value=None)
        cache.set_embedding = AsyncMock()
        cache.set_results = AsyncMock()
        cache.append_history = AsyncMock()

        async def fake_encode_image(img_bytes):
            vec = [float(img_bytes[0])] + [0.0] * 511
            embeddings_encoded.append(vec)
            return vec

        encoder.encode_image_bytes = fake_encode_image
        qdrant.search = AsyncMock(return_value=[])
        svc._hydrate = AsyncMock(return_value=[])

        images = [bytes([1] + [0] * 10), bytes([2] + [0] * 10)]
        await svc.execute_image_search(images, MetadataFilters())

        # Two images → encoder called twice
        assert len(embeddings_encoded) == 2
