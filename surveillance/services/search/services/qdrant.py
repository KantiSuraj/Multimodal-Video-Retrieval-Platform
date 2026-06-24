"""Qdrant client wrapper for the Search Service — the only file allowed
to know about Qdrant SDK types, filter expressions, or search parameters.

Search owns retrieval; Indexing owns persistence.  This module mirrors
the structural contract of indexing/services/qdrant.py but exposes
search operations (ANN retrieval + metadata filtering) instead of
write operations (collection lifecycle + batch upsert).

Invariants enforced here:
- Search never writes vectors (no upsert, no delete, no create).
- Search never creates collections.
- All retrieval parameters (hnsw_ef) are sourced from Settings.
- Qdrant SDK types never escape this module.

Concurrency note: AsyncQdrantClient is safe for concurrent calls from
a single instance.  No additional locking is required.
"""
from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range, SearchParams

from services.search.core.config import Settings
from services.search.core.logging import get_logger
from services.search.models.schemas import MetadataFilters, RetrievedPoint, SearchError, SearchStage

logger = get_logger(__name__)


class QdrantSearchClient:
    """Manages the Qdrant connection and executes ANN retrieval.

    All Qdrant SDK types are contained within this module.  The
    orchestrator (SearchService) speaks only in MetadataFilters and
    RetrievedPoint dataclasses.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncQdrantClient | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """Create the Qdrant client connection."""
        self._client = AsyncQdrantClient(
            host=self._settings.QDRANT_HOST,
            port=self._settings.QDRANT_PORT,
            grpc_port=self._settings.QDRANT_GRPC_PORT,
            api_key=self._settings.QDRANT_API_KEY,
            prefer_grpc=self._settings.QDRANT_USE_GRPC,
            timeout=self._settings.QDRANT_TIMEOUT,
        )
        logger.info(
            "qdrant_search_client_connected",
            host=self._settings.QDRANT_HOST,
            port=self._settings.QDRANT_PORT,
        )

    async def shutdown(self) -> None:
        """Close the Qdrant client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("qdrant_search_client_closed")

    # ── ANN retrieval ─────────────────────────────────────────────────────────

    async def search(
        self,
        query_vector: list[float],
        filters: MetadataFilters,
        top_k: int,
    ) -> list[RetrievedPoint]:
        """Execute ANN retrieval with pre-filtering.

        Filters are applied by Qdrant before ANN traversal — this is
        pre-filtering, not post-processing.  All metadata conditions are
        translated into Qdrant filter expressions here; the orchestrator
        never sees Qdrant SDK types.

        HNSW search-time ef is sourced from settings.SEARCH_HNSW_EF —
        never hardcoded.
        """
        client = self._require_client()
        collection_name = self._settings.QDRANT_COLLECTION_NAME

        qdrant_filter = self._build_filter(filters)

        try:
            results = await client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                search_params=SearchParams(
                    hnsw_ef=self._settings.SEARCH_HNSW_EF,
                    exact=False,
                ),
                with_payload=True,
            )
        except (UnexpectedResponse, Exception) as exc:
            recoverable = _is_transient(exc)
            raise SearchError(
                message=f"Qdrant ANN search failed: {exc}",
                stage=SearchStage.ANN_RETRIEVE,
                recoverable=recoverable,
            ) from exc

        points = [
            RetrievedPoint(
                point_id=str(result.id),
                score=result.score,
                payload=result.payload or {},
            )
            for result in results
        ]

        logger.info(
            "qdrant_search_complete",
            collection=collection_name,
            returned=len(points),
            top_k=top_k,
            hnsw_ef=self._settings.SEARCH_HNSW_EF,
        )
        return points

    # ── Filter construction ───────────────────────────────────────────────────

    @staticmethod
    def _build_filter(filters: MetadataFilters) -> Filter | None:
        """Translate MetadataFilters into a Qdrant Filter expression.

        Uses pre-filtering (must conditions) so Qdrant applies the
        filter before ANN traversal rather than after.  This is the
        correct approach for surveillance use cases where camera/time
        filters dramatically reduce the search space.

        Returns None when no filters are active so Qdrant performs an
        unfiltered search — passing an empty Filter would still incur
        overhead.
        """
        must: list[FieldCondition] = []

        if filters.camera_ids:
            must.append(
                FieldCondition(
                    key="camera_id",
                    match=MatchAny(any=filters.camera_ids),
                )
            )

        if filters.locations:
            must.append(
                FieldCondition(
                    key="location",
                    match=MatchAny(any=filters.locations),
                )
            )

        if filters.start_ms is not None or filters.end_ms is not None:
            range_kwargs: dict[str, Any] = {}
            if filters.start_ms is not None:
                range_kwargs["gte"] = filters.start_ms
            if filters.end_ms is not None:
                range_kwargs["lte"] = filters.end_ms
            must.append(
                FieldCondition(
                    key="timestamp_ms",
                    range=Range(**range_kwargs),
                )
            )

        if filters.labels:
            must.append(
                FieldCondition(
                    key="label",
                    match=MatchAny(any=filters.labels),
                )
            )

        if not must:
            return None

        return Filter(must=must)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantSearchClient not started — call startup() first")
        return self._client


def _is_transient(exc: Exception) -> bool:
    """Heuristic: connection-level and timeout errors are transient.

    Mirrors the identical helper in indexing/services/qdrant.py.
    """
    transient_types = (ConnectionError, TimeoutError, OSError)
    if isinstance(exc, transient_types):
        return True
    if isinstance(exc, UnexpectedResponse):
        status = getattr(exc, "status_code", None)
        if status is not None and status >= 500:
            return True
    return False
