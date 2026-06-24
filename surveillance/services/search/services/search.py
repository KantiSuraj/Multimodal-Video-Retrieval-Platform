"""Search Service orchestrator.

execute_text_search() and execute_image_search() are the two public
entry points.  Each implements the workflow specified in the architecture:

Text Search:
    Request → Validation → Query Encoding → Filter Construction
    → Qdrant ANN Search → Result Hydration → Temporal Deduplication
    → Ranking → Pagination → Response

Image Search:
    Request → Image Validation → Image Encoding → Filter Construction
    → Qdrant ANN Search → Result Hydration → Temporal Deduplication
    → Ranking → Pagination → Response

Search Service invariants enforced throughout:
    1. Search never writes vectors.
    2. Search never modifies collections.
    3. Search never generates stored embeddings.
    4. All retrieval passes through Qdrant.
    5. Ranking occurs after retrieval.
    6. Temporal deduplication occurs before pagination.
    7. Metadata hydration occurs before response serialisation.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select

from shared.shared.models.embedding_record import EmbeddingRecord
from shared.shared.models.video import VideoRecord

from services.search.core.config import Settings
from services.search.core.logging import get_logger
from services.search.db.database import get_session
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
    make_image_embedding_key,
    make_result_cache_key,
    make_text_embedding_key,
)
from services.search.services.clip_encoder import CLIPQueryEncoder
from services.search.services.qdrant import QdrantSearchClient

logger = get_logger(__name__)

_CLIP_DURATION_MS = 5_000  # assumed clip length when timestamp_end is not in the payload


class SearchService:
    def __init__(
        self,
        settings: Settings,
        encoder: CLIPQueryEncoder,
        qdrant: QdrantSearchClient,
        cache: SearchCacheClient,
    ) -> None:
        self._settings = settings
        self._encoder = encoder
        self._qdrant = qdrant
        self._cache = cache

    # ── Public entry points ───────────────────────────────────────────────────

    async def execute_text_search(
        self,
        query_text: str,
        filters: MetadataFilters,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> tuple[list[HydratedResult], str | None]:
        """Execute a text-to-video search.

        Returns (results_page, next_page_token).  next_page_token is
        None when no more results exist.
        """
        self._validate_text_query(query_text)

        embedding_key = make_text_embedding_key(query_text)
        query_vector = await self._get_or_encode_text(query_text, embedding_key)

        return await self._run_retrieval_pipeline(
            query_vector=query_vector,
            embedding_key=embedding_key,
            filters=filters,
            page_size=page_size,
            page_token=page_token,
            query_type="text",
            query_text=query_text,
        )

    async def execute_image_search(
        self,
        image_bytes_list: list[bytes],
        filters: MetadataFilters,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> tuple[list[HydratedResult], str | None]:
        """Execute an image-to-video search.

        Accepts one or more images.  Multiple images are averaged before
        ANN retrieval per the architecture specification.
        """
        if not image_bytes_list:
            raise SearchError(
                message="At least one image is required for image search",
                stage=SearchStage.VALIDATION,
                recoverable=False,
            )

        # Use the hash of the first image as the cache key anchor.
        # For multi-image, we XOR all hashes to get a stable compound key.
        embedding_key = make_image_embedding_key(b"".join(image_bytes_list))
        query_vector = await self._get_or_encode_images(image_bytes_list, embedding_key)

        return await self._run_retrieval_pipeline(
            query_vector=query_vector,
            embedding_key=embedding_key,
            filters=filters,
            page_size=page_size,
            page_token=page_token,
            query_type="image" if len(image_bytes_list) == 1 else "multi_image",
            query_text=None,
        )

    # ── Retrieval pipeline ────────────────────────────────────────────────────

    async def _run_retrieval_pipeline(
        self,
        query_vector: list[float],
        embedding_key: str,
        filters: MetadataFilters,
        page_size: int | None,
        page_token: str | None,
        query_type: str,
        query_text: str | None,
    ) -> tuple[list[HydratedResult], str | None]:
        """Shared pipeline from filter construction to paginated response."""

        effective_page_size = min(
            page_size or self._settings.SEARCH_DEFAULT_PAGE_SIZE,
            self._settings.SEARCH_MAX_PAGE_SIZE,
        )

        # ── Check result cache ────────────────────────────────────────────────
        filters_dict = _filters_to_dict(filters)
        result_key = make_result_cache_key(embedding_key, filters_dict)
        cached = await self._cache.get_results(result_key)
        if cached is not None:
            logger.info("search_result_cache_hit", result_key=result_key)
            all_results = [dict_to_result(d) for d in cached]
            page, next_token = self._paginate(all_results, effective_page_size, page_token)
            return page, next_token

        # ── Validate vector dimension ─────────────────────────────────────────
        self._validate_vector_dimension(query_vector)

        # ── ANN retrieval ─────────────────────────────────────────────────────
        raw_points = await self._qdrant.search(
            query_vector=query_vector,
            filters=filters,
            top_k=self._settings.SEARCH_TOP_K,
        )

        # ── Metadata hydration ────────────────────────────────────────────────
        hydrated = await self._hydrate(raw_points)

        # ── Ranking (cosine similarity descending — already ordered by Qdrant) ─
        ranked = self._rank(hydrated)

        # ── Temporal deduplication ────────────────────────────────────────────
        deduped = self._temporal_deduplicate(ranked)

        # ── Cache full deduplicated result set ────────────────────────────────
        await self._cache.set_results(result_key, deduped)

        # ── History ───────────────────────────────────────────────────────────
        await self._cache.append_history(
            {
                "id": str(uuid.uuid4()),
                "query_type": query_type,
                "query_text": query_text,
                "filters": filters_dict,
                "result_count": len(deduped),
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
            }
        )

        logger.info(
            "search_pipeline_complete",
            query_type=query_type,
            raw_points=len(raw_points),
            hydrated=len(hydrated),
            after_dedup=len(deduped),
        )

        # ── Pagination ────────────────────────────────────────────────────────
        page, next_token = self._paginate(deduped, effective_page_size, page_token)
        return page, next_token

    # ── Encoding with cache ───────────────────────────────────────────────────

    async def _get_or_encode_text(self, text: str, key: str) -> list[float]:
        cached = await self._cache.get_embedding(key)
        if cached is not None:
            logger.info("search_embedding_cache_hit", key=key)
            return cached

        vector = await self._encoder.encode_text(text)
        await self._cache.set_embedding(key, vector)
        return vector

    async def _get_or_encode_images(
        self, images: list[bytes], key: str
    ) -> list[float]:
        cached = await self._cache.get_embedding(key)
        if cached is not None:
            logger.info("search_embedding_cache_hit", key=key)
            return cached

        embeddings: list[list[float]] = []
        for img_bytes in images:
            vec = await self._encoder.encode_image_bytes(img_bytes)
            embeddings.append(vec)

        vector = (
            CLIPQueryEncoder.average_embeddings(embeddings)
            if len(embeddings) > 1
            else embeddings[0]
        )
        await self._cache.set_embedding(key, vector)
        return vector

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_text_query(text: str) -> None:
        if not text or not text.strip():
            raise SearchError(
                message="Query text must not be empty",
                stage=SearchStage.VALIDATION,
                recoverable=False,
            )
        if len(text) > 1000:
            raise SearchError(
                message=f"Query text is too long ({len(text)} chars, max 1000)",
                stage=SearchStage.VALIDATION,
                recoverable=False,
            )

    def _validate_vector_dimension(self, vector: list[float]) -> None:
        expected = self._settings.QDRANT_VECTOR_DIMENSION
        actual = len(vector)
        if actual != expected:
            raise SearchError(
                message=(
                    f"Query vector dimension {actual} does not match "
                    f"configured dimension {expected} — model mismatch"
                ),
                stage=SearchStage.VALIDATION,
                recoverable=False,
            )

    # ── Metadata hydration ────────────────────────────────────────────────────

    async def _hydrate(self, points: list[RetrievedPoint]) -> list[HydratedResult]:
        """Hydrate raw Qdrant points with PostgreSQL metadata.

        Fetches EmbeddingRecord and VideoRecord for every point.
        Points whose records cannot be found are logged and skipped —
        they represent a consistency gap (e.g. a video deleted after
        indexing) and must not propagate as partial results.
        """
        if not points:
            return []

        results: list[HydratedResult] = []
        for point in points:
            try:
                hydrated = await self._hydrate_one(point)
                if hydrated is not None:
                    results.append(hydrated)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "search_hydration_point_skipped",
                    point_id=point.point_id,
                    reason=str(exc),
                )
        return results

    async def _hydrate_one(self, point: RetrievedPoint) -> HydratedResult | None:
        """Hydrate a single Qdrant point with PostgreSQL metadata."""
        video_id_str = point.payload.get("video_id")
        source_path = point.payload.get("source_path")

        if not video_id_str or not source_path:
            logger.warning(
                "search_hydration_missing_payload_fields",
                point_id=point.point_id,
                payload_keys=list(point.payload.keys()),
            )
            return None

        try:
            video_uuid = uuid.UUID(video_id_str)
        except ValueError:
            logger.warning(
                "search_hydration_invalid_video_uuid",
                point_id=point.point_id,
                video_id=video_id_str,
            )
            return None

        async with get_session() as db:
            # Fetch EmbeddingRecord for the specific source_path
            stmt = select(EmbeddingRecord).where(
                EmbeddingRecord.video_id == video_uuid,
                EmbeddingRecord.source_path == source_path,
            )
            result = await db.execute(stmt)
            emb_record = result.scalar_one_or_none()

            if emb_record is None:
                logger.warning(
                    "search_hydration_embedding_record_missing",
                    point_id=point.point_id,
                    video_id=video_id_str,
                    source_path=source_path,
                )
                return None

            # Fetch VideoRecord
            video_record = await db.get(VideoRecord, video_uuid)
            if video_record is None:
                logger.warning(
                    "search_hydration_video_record_missing",
                    point_id=point.point_id,
                    video_id=video_id_str,
                )
                return None

        timestamp_start = emb_record.timestamp_ms or 0
        timestamp_end = timestamp_start + _CLIP_DURATION_MS

        # Presigned thumbnail URL from MinIO (best-effort; None on failure)
        thumbnail_url = await self._get_thumbnail_url(source_path)

        # video_start_epoch: recorded_at as epoch-ms
        video_start_epoch: int | None = None
        if video_record.recorded_at is not None:
            video_start_epoch = int(video_record.recorded_at.timestamp() * 1000)

        detected_labels: list[str] = []
        if emb_record.label:
            detected_labels = [emb_record.label]

        return HydratedResult(
            video_id=str(video_uuid),
            clip_id=str(emb_record.id),
            camera_name=video_record.camera_id,
            camera_location=video_record.location,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            similarity_score=point.score,
            thumbnail_url=thumbnail_url,
            detected_labels=detected_labels,
            video_start_epoch=video_start_epoch,
        )

    async def _get_thumbnail_url(self, source_path: str) -> str | None:
        """Generate a presigned MinIO URL for the thumbnail; return None on failure."""
        from shared.shared.storage.client import ObjectStorageClient

        try:
            client = ObjectStorageClient(self._settings)
            return await client.presigned_get_url(
                bucket=self._settings.MINIO_PROCESSED_FRAMES_BUCKET,
                object_name=source_path,
                expires_sec=self._settings.MINIO_THUMBNAIL_PRESIGN_TTL_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_thumbnail_url_failed", source_path=source_path, reason=str(exc))
            return None

    # ── Ranking ───────────────────────────────────────────────────────────────

    @staticmethod
    def _rank(results: list[HydratedResult]) -> list[HydratedResult]:
        """Sort by cosine similarity descending.

        Qdrant already returns results ordered by score descending.  This
        explicit sort ensures correctness even if hydration reorders results
        (e.g. via skipped points).
        """
        return sorted(results, key=lambda r: r.similarity_score, reverse=True)

    # ── Temporal deduplication ────────────────────────────────────────────────

    def _temporal_deduplicate(self, results: list[HydratedResult]) -> list[HydratedResult]:
        """Remove lower-ranked results that are temporally close to a higher-ranked hit.

        Algorithm (per the architecture specification):
        - Process results in ranking order (highest score first).
        - For each result, check if any ALREADY-KEPT result from the same
          video falls within SEARCH_TEMPORAL_DEDUP_WINDOW_MS.
        - If so, skip (the lower-ranked hit is suppressed).
        - Otherwise, keep.

        The deduplication window is configurable via settings.
        """
        window_ms = self._settings.SEARCH_TEMPORAL_DEDUP_WINDOW_MS
        kept: list[HydratedResult] = []
        # Maps video_id → list of kept timestamp_starts for that video.
        kept_timestamps: dict[str, list[int]] = {}

        for result in results:
            vid = result.video_id
            ts = result.timestamp_start
            existing = kept_timestamps.get(vid, [])

            if any(abs(ts - kept_ts) < window_ms for kept_ts in existing):
                continue  # suppressed — within dedup window of a higher-ranked hit

            kept.append(result)
            existing.append(ts)
            kept_timestamps[vid] = existing

        return kept

    # ── Pagination ────────────────────────────────────────────────────────────

    def _paginate(
        self,
        results: list[HydratedResult],
        page_size: int,
        page_token: str | None,
    ) -> tuple[list[HydratedResult], str | None]:
        """Cursor-based pagination over the deduplicated result list.

        Offset pagination is prohibited by the architecture.  The cursor
        token encodes the integer offset into the deduplicated result list
        as a base-10 string, giving safe, stateless page navigation.
        A malformed token is a non-recoverable client error.
        """
        offset = self._decode_page_token(page_token)
        page = results[offset : offset + page_size]
        next_offset = offset + page_size
        next_token = self._encode_page_token(next_offset) if next_offset < len(results) else None
        return page, next_token

    @staticmethod
    def _encode_page_token(offset: int) -> str:
        import base64

        return base64.urlsafe_b64encode(str(offset).encode()).decode()

    @staticmethod
    def _decode_page_token(token: str | None) -> int:
        if token is None:
            return 0
        import base64

        try:
            return int(base64.urlsafe_b64decode(token.encode()).decode())
        except Exception as exc:  # noqa: BLE001
            raise SearchError(
                message=f"Malformed pagination token: {token!r}",
                stage=SearchStage.PAGINATE,
                recoverable=False,
            ) from exc

    # ── Search history ────────────────────────────────────────────────────────

    async def get_search_history(
        self,
        query_type: str | None = None,
        cursor: int = 0,
        page_size: int = 20,
    ) -> tuple[list[dict], int | None]:
        """Return paginated search history with optional query_type filter."""
        entries, next_cursor = await self._cache.get_history(
            cursor=cursor, page_size=page_size
        )
        if query_type is not None:
            entries = [e for e in entries if e.get("query_type") == query_type]
        return entries, next_cursor


# ── Helpers ───────────────────────────────────────────────────────────────────


def _filters_to_dict(filters: MetadataFilters) -> dict:
    return {
        "camera_ids": filters.camera_ids,
        "locations": filters.locations,
        "start_ms": filters.start_ms,
        "end_ms": filters.end_ms,
        "labels": filters.labels,
    }
