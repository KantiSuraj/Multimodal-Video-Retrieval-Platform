"""Qdrant client wrapper — the only file allowed to know about Qdrant
SDK types, collection schemas, or HNSW parameters.

Indexing owns vector persistence; this module is the sole gateway to
Qdrant.  It exposes collection lifecycle and batch upsert operations.
It intentionally contains zero retrieval, search, or ranking behaviour —
those belong exclusively to the Search Service.

Concurrency note: the qdrant_client AsyncQdrantClient is safe for
concurrent calls from a single instance.  No additional locking is
needed beyond what aio_pika's prefetch already provides.
"""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)

from services.indexing.core.config import Settings
from services.indexing.core.logging import get_logger
from services.indexing.models.schemas import IndexingError, IndexingStage, QdrantPoint

logger = get_logger(__name__)

_DISTANCE_MAP = {
    "Cosine": Distance.COSINE,
    "Euclid": Distance.EUCLID,
    "Dot": Distance.DOT,
}


class QdrantService:
    """Manages Qdrant collection lifecycle and batch point upserts.

    All Qdrant SDK types are contained within this module.  The
    orchestrator (IndexingService) never touches qdrant_client types
    directly — it speaks only in QdrantPoint dataclasses.
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
            "qdrant_client_connected",
            host=self._settings.QDRANT_HOST,
            port=self._settings.QDRANT_PORT,
        )

    async def shutdown(self) -> None:
        """Close the Qdrant client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("qdrant_client_closed")

    # ── Collection lifecycle ──────────────────────────────────────────────────

    async def ensure_collection(self) -> None:
        """Idempotent collection creation.

        Checks whether the configured collection already exists.  If it
        does, validates that the vector dimension and distance metric
        match the current configuration.  If it does not, creates it
        with explicit HNSW parameters.

        Safe to call on every startup and before every indexing run.
        """
        client = self._require_client()
        collection_name = self._settings.QDRANT_COLLECTION_NAME

        exists = await client.collection_exists(collection_name)

        if exists:
            await self._validate_collection(client, collection_name)
            logger.info(
                "qdrant_collection_validated",
                collection=collection_name,
            )
            return

        distance = self._resolve_distance()

        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self._settings.QDRANT_VECTOR_DIMENSION,
                distance=distance,
            ),
            hnsw_config=HnswConfigDiff(
                m=self._settings.QDRANT_HNSW_M,
                ef_construct=self._settings.QDRANT_HNSW_EF_CONSTRUCT,
            ),
        )
        logger.info(
            "qdrant_collection_created",
            collection=collection_name,
            dimension=self._settings.QDRANT_VECTOR_DIMENSION,
            distance=self._settings.QDRANT_DISTANCE_METRIC,
            hnsw_m=self._settings.QDRANT_HNSW_M,
            hnsw_ef_construct=self._settings.QDRANT_HNSW_EF_CONSTRUCT,
        )

    async def _validate_collection(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
    ) -> None:
        """Verify that an existing collection matches the configured schema.

        Dimension/distance mismatches are non-recoverable — the
        collection was created with different parameters and cannot be
        reconciled at runtime.
        """
        info = await client.get_collection(collection_name)
        vectors_config = info.config.params.vectors

        # vectors_config can be a VectorParams (single unnamed vector)
        # or a dict of named vectors.  We only use unnamed vectors.
        if isinstance(vectors_config, dict):
            raise IndexingError(
                message=(
                    f"Collection '{collection_name}' uses named vectors, "
                    "but indexing expects a single unnamed vector configuration"
                ),
                stage=IndexingStage.COLLECTION_INIT,
                recoverable=False,
            )

        expected_dim = self._settings.QDRANT_VECTOR_DIMENSION
        actual_dim = vectors_config.size
        if actual_dim != expected_dim:
            raise IndexingError(
                message=(
                    f"Collection '{collection_name}' has dimension {actual_dim}, "
                    f"expected {expected_dim}"
                ),
                stage=IndexingStage.COLLECTION_INIT,
                recoverable=False,
            )

        expected_distance = self._resolve_distance()
        actual_distance = vectors_config.distance
        if actual_distance != expected_distance:
            raise IndexingError(
                message=(
                    f"Collection '{collection_name}' has distance {actual_distance}, "
                    f"expected {expected_distance}"
                ),
                stage=IndexingStage.COLLECTION_INIT,
                recoverable=False,
            )

    # ── Batch upsert ──────────────────────────────────────────────────────────

    async def batch_upsert(self, points: list[QdrantPoint]) -> None:
        """Upsert points in configurable batches.

        Never upserts one-at-a-time.  Splits the full point list into
        chunks of QDRANT_UPSERT_BATCH_SIZE and issues one upsert per
        chunk, minimising network round-trips.

        Point IDs are deterministic, so duplicate deliveries overwrite
        the same Qdrant points — no duplicates are created.
        """
        if not points:
            return

        client = self._require_client()
        collection_name = self._settings.QDRANT_COLLECTION_NAME
        batch_size = self._settings.QDRANT_UPSERT_BATCH_SIZE

        structs = [
            PointStruct(
                id=p.point_id,
                vector=p.vector,
                payload=p.payload,
            )
            for p in points
        ]

        for i in range(0, len(structs), batch_size):
            batch = structs[i : i + batch_size]
            try:
                await client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True,
                )
            except (UnexpectedResponse, Exception) as exc:
                # Qdrant unavailable / timeout → recoverable.
                # Malformed points → non-recoverable (should not happen
                # because we validate vectors before reaching here).
                recoverable = _is_transient(exc)
                raise IndexingError(
                    message=f"Qdrant upsert failed (batch {i // batch_size}): {exc}",
                    stage=IndexingStage.UPSERT,
                    recoverable=recoverable,
                ) from exc

        logger.info(
            "qdrant_batch_upsert_complete",
            collection=collection_name,
            total_points=len(points),
            batch_count=(len(points) + batch_size - 1) // batch_size,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantService not started — call startup() first")
        return self._client

    def _resolve_distance(self) -> Distance:
        metric = self._settings.QDRANT_DISTANCE_METRIC
        distance = _DISTANCE_MAP.get(metric)
        if distance is None:
            raise IndexingError(
                message=f"Unknown distance metric: '{metric}'. Expected one of {list(_DISTANCE_MAP.keys())}",
                stage=IndexingStage.COLLECTION_INIT,
                recoverable=False,
            )
        return distance


def _is_transient(exc: Exception) -> bool:
    """Heuristic: connection-level and timeout errors are transient."""
    transient_types = (ConnectionError, TimeoutError, OSError)
    if isinstance(exc, transient_types):
        return True
    # qdrant_client wraps HTTP errors in UnexpectedResponse
    if isinstance(exc, UnexpectedResponse):
        status = getattr(exc, "status_code", None)
        if status is not None and status >= 500:
            return True
    return False
