"""Orchestrator. process_embeddings() is the entry point every message reaches.

Consumes shared.events.embeddings_ready.EmbeddingsReadyEvent (the
shared/ contract: video_id: str, model_name: str, embeddings:
list[EmbeddingRecord], each EmbeddingRecord carrying kind, source_path,
vector, timestamp_ms, label).

For every embedding in the event this service:
1. Validates the vector dimension against the configured schema.
2. Generates a deterministic UUID5 point ID from (video_id, source_path)
   so that duplicate event delivery overwrites the same Qdrant point.
3. Transforms the embedding into a QdrantPoint with a rich payload
   (video_id, kind, source_path, model_name, timestamp_ms, label).
4. Batch-upserts all points into Qdrant.
5. Updates the shared EmbeddingRecord rows with qdrant_point_id and
   qdrant_collection — these columns were left null by the embedding
   service and are owned by indexing.
6. Transitions VideoRecord.status to INDEXED.

Idempotency mirrors embedding's pattern exactly:
- VideoStatus.INDEXED is checked the same way embedding checks
  VideoStatus.EMBEDDED.
- Point IDs are deterministic, so re-processing the same event
  overwrites the same Qdrant points rather than creating duplicates.
- EmbeddingRecord.qdrant_point_id updates are idempotent (UPDATE WHERE).

Persistence happens before the status update, matching the documented
data flow: Validation → Transformation → Qdrant Upsert → Metadata
Persist → Status Update.
"""
from __future__ import annotations

import uuid

from sqlalchemy import update

from shared.shared.events.embeddings_ready import EmbeddingsReadyEvent
from shared.shared.events.embeddings_ready import EmbeddingRecord as EmbeddingPayload
from shared.shared.models.embedding_record import EmbeddingRecord
from shared.shared.models.video import VideoRecord, VideoStatus

from services.indexing.core.config import Settings
from services.indexing.core.logging import get_logger
from services.indexing.db.database import get_session
from services.indexing.models.schemas import (
    IndexingError,
    IndexingStage,
    QdrantPoint,
)
from services.indexing.services.qdrant import QdrantService

logger = get_logger(__name__)

# Deterministic namespace for UUID5 point IDs.
# Using a fixed namespace means the same (video_id, source_path) always
# produces the same point ID across restarts and redeliveries.
_POINT_ID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


class IndexingService:
    def __init__(
        self,
        settings: Settings,
        qdrant: QdrantService,
    ):
        self._settings = settings
        self._qdrant = qdrant

    async def process_embeddings(self, event: EmbeddingsReadyEvent) -> None:
        video_id = event.video_id
        is_final_batch = (event.batch_index == event.total_batches - 1)
        is_first_batch = (event.batch_index == 0)

        # Only apply the idempotency guard on the FIRST batch (batch_index=0).
        # Intermediate and final batches must always be upserted regardless of
        # video status — the video is already PROCESSING when they arrive.
        # Applying the guard on every batch would skip batches 1..N-1 because
        # as soon as batch 0 is processed the video is still PROCESSING (or
        # INDEXED if a prior full run completed), causing partial indexing.
        if is_first_batch and await self._already_indexed(video_id):
            logger.info("indexing_already_indexed_skipped", video_id=video_id)
            return

        # Only transition to PROCESSING on the first batch to avoid
        # redundant DB writes on every intermediate batch.
        if is_first_batch:
            await self._mark_status(video_id, VideoStatus.PROCESSING)

        logger.info(
            "indexing_batch_progress",
            video_id=video_id,
            batch_index=event.batch_index,
            total_batches=event.total_batches,
            embeddings_in_batch=len(event.embeddings),
        )

        try:
            self._validate_embeddings(event)
            points = self._transform_to_points(event)
            await self._qdrant.ensure_collection()
            await self._qdrant.batch_upsert(points)
            await self._update_embedding_records(event, points)

            # Only mark INDEXED after the final batch has been persisted.
            # Marking it earlier causes all subsequent batches to hit
            # _already_indexed() and be silently dropped.
            if is_final_batch:
                await self._mark_status(video_id, VideoStatus.INDEXED)
                logger.info(
                    "indexing_complete",
                    video_id=video_id,
                    points_indexed=len(points),
                    total_batches=event.total_batches,
                )
            else:
                logger.debug(
                    "indexing_batch_upserted",
                    video_id=video_id,
                    batch_index=event.batch_index,
                    total_batches=event.total_batches,
                    points_upserted=len(points),
                )
        except IndexingError as exc:
            if exc.recoverable:
                logger.warning(
                    "indexing_recoverable_failure",
                    video_id=video_id,
                    stage=exc.stage.value,
                    reason=exc.message,
                )
                raise
            logger.error(
                "indexing_permanent_failure",
                video_id=video_id,
                stage=exc.stage.value,
                reason=exc.message,
            )
            await self._mark_status(video_id, VideoStatus.FAILED, error_message=exc.message)

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_embeddings(self, event: EmbeddingsReadyEvent) -> None:
        """Validate every embedding vector in the event.

        Dimension mismatches are non-recoverable — the embedding model
        produced vectors incompatible with the configured collection.
        Invalid vectors must never reach Qdrant.
        """
        expected_dim = self._settings.QDRANT_VECTOR_DIMENSION

        if not event.embeddings:
            raise IndexingError(
                message=f"EmbeddingsReadyEvent for video {event.video_id} contains no embeddings",
                stage=IndexingStage.VALIDATE,
                recoverable=False,
            )

        for idx, emb in enumerate(event.embeddings):
            actual_dim = len(emb.vector)
            if actual_dim != expected_dim:
                raise IndexingError(
                    message=(
                        f"Embedding {idx} (kind={emb.kind}, source={emb.source_path}) "
                        f"has dimension {actual_dim}, expected {expected_dim}"
                    ),
                    stage=IndexingStage.VALIDATE,
                    recoverable=False,
                )

            if not emb.vector:
                raise IndexingError(
                    message=(
                        f"Embedding {idx} (kind={emb.kind}, source={emb.source_path}) "
                        "has an empty vector"
                    ),
                    stage=IndexingStage.VALIDATE,
                    recoverable=False,
                )

    # ── Transformation ────────────────────────────────────────────────────────

    def _transform_to_points(self, event: EmbeddingsReadyEvent) -> list[QdrantPoint]:
        """Transform embeddings into Qdrant-ready points.

        Point IDs are deterministic UUID5 values derived from
        (video_id, source_path).  This ensures:
        - Duplicate event delivery overwrites, never duplicates.
        - Worker crashes and retries are safe.
        - The same embedding always maps to the same Qdrant point.
        """
        points: list[QdrantPoint] = []
        for emb in event.embeddings:
            point_id = self._deterministic_point_id(event.video_id, emb.source_path)
            payload = self._build_payload(event, emb)
            points.append(
                QdrantPoint(
                    point_id=point_id,
                    vector=emb.vector,
                    payload=payload,
                )
            )
        return points

    @staticmethod
    def _deterministic_point_id(video_id: str, source_path: str) -> str:
        """Generate a deterministic UUID5 from video_id + source_path.

        Using UUID5 (SHA-1-based) ensures the same inputs always produce
        the same output, making upserts idempotent across redeliveries.
        """
        name = f"{video_id}:{source_path}"
        return str(uuid.uuid5(_POINT_ID_NAMESPACE, name))

    @staticmethod
    def _build_payload(event: EmbeddingsReadyEvent, emb: EmbeddingPayload) -> dict:
        """Build the Qdrant point payload.

        The payload carries all metadata needed for post-retrieval
        filtering and display by the Search Service.  The payload is
        intentionally flat — no nested objects — for efficient Qdrant
        payload indexing.
        """
        payload: dict = {
            "video_id": event.video_id,
            "kind": emb.kind,
            "source_path": emb.source_path,
            "model_name": event.model_name,
        }
        if emb.timestamp_ms is not None:
            payload["timestamp_ms"] = emb.timestamp_ms
        if emb.label is not None:
            payload["label"] = emb.label
        return payload

    # ── Metadata persistence ──────────────────────────────────────────────────

    async def _update_embedding_records(
        self,
        event: EmbeddingsReadyEvent,
        points: list[QdrantPoint],
    ) -> None:
        """Update EmbeddingRecord rows with Qdrant point references.

        The embedding service created EmbeddingRecord rows with
        qdrant_point_id=NULL and qdrant_collection=NULL.  Indexing owns
        these columns — it fills them after successful Qdrant upsert.

        Updates are matched by (video_id, source_path), making this
        idempotent: re-indexing the same video overwrites the same rows
        with the same deterministic point IDs.
        """
        collection_name = self._settings.QDRANT_COLLECTION_NAME
        video_uuid = uuid.UUID(event.video_id)

        try:
            async with get_session() as db:
                for emb, point in zip(event.embeddings, points):
                    await db.execute(
                        update(EmbeddingRecord)
                        .where(
                            EmbeddingRecord.video_id == video_uuid,
                            EmbeddingRecord.source_path == emb.source_path,
                        )
                        .values(
                            qdrant_point_id=point.point_id,
                            qdrant_collection=collection_name,
                        )
                    )
        except Exception as exc:
            raise IndexingError(
                message=f"Failed to update embedding records: {exc}",
                stage=IndexingStage.METADATA_PERSIST,
                recoverable=True,
            ) from exc

    # ── Status management ─────────────────────────────────────────────────────

    async def _already_indexed(self, video_id: str) -> bool:
        """Check if this video has already been indexed."""
        try:
            video_uuid = uuid.UUID(video_id)
        except ValueError:
            logger.warning("indexing_invalid_video_id", video_id=video_id)
            return False

        async with get_session() as db:
            record = await db.get(VideoRecord, video_uuid)
            if record is None:
                logger.warning("indexing_video_record_missing", video_id=video_id)
                return False
            return record.status == VideoStatus.INDEXED

    async def _mark_status(
        self, video_id: str, status: VideoStatus, error_message: str | None = None
    ) -> None:
        """Transition VideoRecord.status.

        Mirrors the exact pattern from embedding's _mark_status.
        """
        try:
            video_uuid = uuid.UUID(video_id)
        except ValueError:
            logger.warning(
                "indexing_invalid_video_id_on_status_update",
                video_id=video_id,
                status=status.value,
            )
            return

        async with get_session() as db:
            record = await db.get(VideoRecord, video_uuid)
            if record is None:
                logger.warning(
                    "indexing_video_record_missing_on_status_update",
                    video_id=video_id,
                    status=status.value,
                )
                return
            record.status = status
            if error_message is not None:
                record.error_message = error_message
