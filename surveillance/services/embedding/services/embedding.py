"""Orchestrator. process_detections() is the entry point every message reaches.

Consumes shared.events.detection_complete.DetectionCompleteEvent (the
shared/ contract: video_id: uuid.UUID, frames: list[FrameDetections], each
FrameDetections carrying its own detections: list[Detection]). For every
frame this builds one "frame" artifact (frame.frame_path, read from
settings.MINIO_PROCESSED_FRAMES_BUCKET — the same bucket preprocessing
wrote to and detection already reads from) and, for every detection inside
that frame that produced a crop, one "crop" artifact (detection.crop_path,
read from detection.crop_bucket if the event carried one, else falling
back to settings.MINIO_DETECTION_CROPS_BUCKET).

Each artifact is fetched from MinIO, embedded with CLIP, L2-normalised
(services/clip_model.py owns that math), and assembled into two parallel
outputs:
- shared.models.embedding_record.EmbeddingRecord rows — metadata only, no
  vector column. qdrant_point_id/qdrant_collection are left null: per the
  architecture doc those are written later by indexing once it has
  upserted into Qdrant ("Created by embedding, updated by indexing (adds
  Qdrant ID)").
- shared.events.embeddings_ready.EmbeddingRecord payload entries — these DO
  carry the vector — for the outgoing EmbeddingsReadyEvent.

Idempotency mirrors detection's pattern exactly:
- VideoStatus.EMBEDDED (added to shared/models/video.py — additive) is
  checked the same way detection checks VideoStatus.DETECTED.
- EmbeddingRecord rows for the video are cleared before re-inserting, since
  a plain INSERT — unlike MinIO put_object — is not idempotent on retry.

Persistence happens before the event is published, matching the documented
data flow: Artifact Retrieval -> Embedding Model -> Embedding Assembly ->
Persistence -> EmbeddingsReadyEvent -> Publisher.
"""
from __future__ import annotations

import os
import shutil
import uuid

from minio.error import S3Error
from sqlalchemy import delete

from shared.shared.events.detection_complete import DetectionCompleteEvent
from shared.shared.events.embeddings_ready import EmbeddingRecord as EmbeddingPayload
from shared.shared.events.embeddings_ready import EmbeddingsReadyEvent
from shared.shared.models.embedding_record import EmbeddingRecord
from shared.shared.models.video import VideoRecord, VideoStatus
from shared.shared.storage.client import _PERMANENT_S3_ERROR_CODES

from services.embedding.core.config import Settings
from services.embedding.core.logging import get_logger
from services.embedding.db.database import get_session
from services.embedding.models.schemas import (
    ArtifactRef,
    EmbeddingError,
    EmbeddingStage,
    PersistedEmbedding,
)
from services.embedding.services.clip_model import CLIPEmbedder
from services.embedding.services.queue import EmbeddingPublisher
from services.embedding.services.storage import EmbeddingStorageService

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(
        self,
        settings: Settings,
        model: CLIPEmbedder,
        storage: EmbeddingStorageService,
        publisher: EmbeddingPublisher,
    ):
        self._settings = settings
        self._model = model
        self._storage = storage
        self._publisher = publisher

    async def process_detections(self, event: DetectionCompleteEvent) -> None:
        video_id = event.video_id

        if await self._already_processed(video_id):
            logger.info("embedding_already_processed_skipped", video_id=str(video_id))
            return

        await self._mark_status(video_id, VideoStatus.PROCESSING)
        # Clear any partial rows from a prior crashed/redelivered attempt
        # before writing new ones — closes the duplicate-row gap, since
        # unlike MinIO put_object, a plain INSERT is not idempotent on retry.
        await self._clear_existing_records(video_id)

        try:
            artifacts = self._collect_artifacts(event)
            if not artifacts:
                logger.warning(
                    "embedding_no_artifacts_skipped",
                    video_id=str(video_id),
                )
                await self._mark_status(video_id, VideoStatus.FAILED,
                                        error_message="DetectionCompleteEvent carried no embeddable artifacts")
                return
            persisted = await self._embed_artifacts(video_id, artifacts)
            await self._write_embedding_rows(video_id, persisted)
            await self._publish_in_batches(video_id, persisted)
            await self._mark_status(video_id, VideoStatus.EMBEDDED)
        except EmbeddingError as exc:
            if exc.recoverable:
                logger.warning(
                    "embedding_recoverable_failure",
                    video_id=str(video_id),
                    stage=exc.stage.value,
                    reason=exc.message,
                )
                raise
            await self._mark_status(video_id, VideoStatus.FAILED, error_message=exc.message)
        except Exception as exc:  # noqa: BLE001
            # Non-EmbeddingError exceptions (e.g. AMQP publish failure, DB
            # error on _mark_status) must NOT leave the video stuck at
            # PROCESSING forever.  Wrap and mark FAILED so operators can
            # identify and replay the video.
            logger.error(
                "embedding_unexpected_failure",
                video_id=str(video_id),
                reason=str(exc),
                exc_info=True,
            )
            await self._mark_status(
                video_id,
                VideoStatus.FAILED,
                error_message=f"Unexpected error: {exc}",
            )
            raise EmbeddingError(
                message=str(exc),
                stage=EmbeddingStage.PUBLISH,
                recoverable=True,   # retry — transient publish failures are recoverable
            ) from exc
        finally:
            self._cleanup_tmp_files(video_id)

    def _collect_artifacts(self, event: DetectionCompleteEvent) -> list[ArtifactRef]:
        artifacts: list[ArtifactRef] = []
        for frame in event.frames:
            artifacts.append(
                ArtifactRef(
                    kind="frame",
                    source_path=frame.frame_path,
                    source_bucket=self._settings.MINIO_PROCESSED_FRAMES_BUCKET,
                    timestamp_ms=frame.timestamp_ms,
                    label=None,
                )
            )
            for detection in frame.detections:
                if not detection.crop_path:
                    continue
                artifacts.append(
                    ArtifactRef(
                        kind="crop",
                        source_path=detection.crop_path,
                        source_bucket=detection.crop_bucket
                        or self._settings.MINIO_DETECTION_CROPS_BUCKET,
                        timestamp_ms=frame.timestamp_ms,
                        label=detection.label,
                        detection_id=detection.detection_id,
                    )
                )
        return artifacts

    async def _embed_artifacts(
        self, video_id: uuid.UUID, artifacts: list[ArtifactRef]
    ) -> list[PersistedEmbedding]:
        persisted: list[PersistedEmbedding] = []
        for idx, artifact in enumerate(artifacts):
            local_path = await self._fetch_and_write_artifact(video_id, idx, artifact)
            vector = await self._model.embed_image(local_path)
            persisted.append(PersistedEmbedding(artifact=artifact, vector=vector))
        return persisted

    async def _fetch_and_write_artifact(
        self, video_id: uuid.UUID, idx: int, artifact: ArtifactRef
    ) -> str:
        try:
            data = await self._storage.fetch_artifact(artifact.source_bucket, artifact.source_path)
        except S3Error as exc:
            # Distinguish permanent S3 errors (e.g. NoSuchKey — the object was
            # never written by preprocessing/detection) from transient ones.
            # Permanent errors must NOT be requeued: the key will never appear
            # on its own, so marking recoverable=True would cause an infinite
            # nack-redeliver loop.  Route to DLQ instead (recoverable=False).
            is_permanent = exc.code in _PERMANENT_S3_ERROR_CODES
            if is_permanent:
                logger.error(
                    "embedding_artifact_missing",
                    video_id=str(video_id),
                    bucket=artifact.source_bucket,
                    path=artifact.source_path,
                    s3_code=exc.code,
                )
            raise EmbeddingError(
                message=f"Failed to fetch artifact {artifact.source_path}: {exc}",
                stage=EmbeddingStage.FETCH_ARTIFACT,
                recoverable=not is_permanent,  # permanent S3 error → DLQ, not requeue
            ) from exc
        except Exception as exc:  # noqa: BLE001
            # Non-S3 exceptions (network timeout, connection refused) are
            # transient and safe to retry via requeue.
            raise EmbeddingError(
                message=f"Failed to fetch artifact {artifact.source_path}: {exc}",
                stage=EmbeddingStage.FETCH_ARTIFACT,
                recoverable=True,
            ) from exc

        tmp_dir = os.path.join(self._settings.EMBEDDING_TMP_DIR, str(video_id))
        os.makedirs(tmp_dir, exist_ok=True)
        local_path = os.path.join(tmp_dir, f"{idx:06d}_{artifact.kind}.jpg")
        with open(local_path, "wb") as f:
            f.write(data)
        return local_path

    async def _write_embedding_rows(
        self, video_id: uuid.UUID, persisted: list[PersistedEmbedding]
    ) -> None:
        if not persisted:
            return
        rows = [
            EmbeddingRecord(
                video_id=video_id,
                kind=p.artifact.kind,
                source_path=p.artifact.source_path,
                model_name=self._settings.CLIP_MODEL_NAME,
                timestamp_ms=p.artifact.timestamp_ms,
                label=p.artifact.label,
                vector_dim=len(p.vector),
            )
            for p in persisted
        ]
        async with get_session() as db:
            for row in rows:
                db.add(row)

    async def _clear_existing_records(self, video_id: uuid.UUID) -> None:
        async with get_session() as db:
            await db.execute(delete(EmbeddingRecord).where(EmbeddingRecord.video_id == video_id))

    def _build_event(
        self,
        video_id: uuid.UUID,
        persisted: list[PersistedEmbedding],
        batch_index: int,
        total_batches: int,
    ) -> EmbeddingsReadyEvent:
        embeddings_payload = [
            EmbeddingPayload(
                kind=p.artifact.kind,
                source_path=p.artifact.source_path,
                vector=p.vector,
                timestamp_ms=p.artifact.timestamp_ms,
                label=p.artifact.label,
            )
            for p in persisted
        ]
        return EmbeddingsReadyEvent(
            video_id=str(video_id),
            model_name=self._settings.CLIP_MODEL_NAME,
            embeddings=embeddings_payload,
            batch_index=batch_index,
            total_batches=total_batches,
        )

    async def _publish_in_batches(
        self, video_id: uuid.UUID, persisted: list[PersistedEmbedding]
    ) -> None:
        """Publish EmbeddingsReadyEvent in chunks to avoid AMQP frame-size overflow.

        Problem: serialising 333 embeddings × 512 floats as JSON produces
        ~4–8 MB which exceeds the aio_pika default frame size (128 KB).
        Solution: split into batches of EMBEDDING_PUBLISH_BATCH_SIZE and
        publish one RabbitMQ message per batch.  Indexing consumes each
        batch independently — idempotency is preserved via deterministic
        Qdrant point IDs derived from (video_id, source_path).

        Any AMQP exception is wrapped as a recoverable EmbeddingError so
        the caller's except clause catches it and marks VideoStatus.FAILED
        instead of leaving the video stuck at PROCESSING.
        """
        batch_size = self._settings.EMBEDDING_PUBLISH_BATCH_SIZE
        batches = [
            persisted[i : i + batch_size]
            for i in range(0, len(persisted), batch_size)
        ]
        logger.info(
            "embedding_publish_batches",
            video_id=str(video_id),
            total=len(persisted),
            batches=len(batches),
            batch_size=batch_size,
        )
        for batch_idx, batch in enumerate(batches):
            try:
                event = self._build_event(
                    video_id,
                    batch,
                    batch_index=batch_idx,
                    total_batches=len(batches),
                )
                await self._publisher.publish_embeddings_ready(event)
                logger.debug(
                    "embedding_batch_published",
                    video_id=str(video_id),
                    batch=batch_idx + 1,
                    of=len(batches),
                    count=len(batch),
                )
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    message=(
                        f"Failed to publish embedding batch {batch_idx + 1}/{len(batches)}"
                        f" for video {video_id}: {exc}"
                    ),
                    stage=EmbeddingStage.PUBLISH,
                    recoverable=True,
                ) from exc

    async def _already_processed(self, video_id: uuid.UUID) -> bool:
        async with get_session() as db:
            record = await db.get(VideoRecord, video_id)
            if record is None:
                logger.warning("embedding_video_record_missing", video_id=str(video_id))
                return False
            return record.status in (VideoStatus.EMBEDDED, VideoStatus.INDEXED)

    async def _mark_status(
        self, video_id: uuid.UUID, status: VideoStatus, error_message: str | None = None
    ) -> None:
        async with get_session() as db:
            record = await db.get(VideoRecord, video_id)
            if record is None:
                logger.warning(
                    "embedding_video_record_missing_on_status_update",
                    video_id=str(video_id),
                    status=status.value,
                )
                return
            record.status = status
            if error_message is not None:
                record.error_message = error_message

    def _cleanup_tmp_files(self, video_id: uuid.UUID) -> None:
        tmp_dir = os.path.join(self._settings.EMBEDDING_TMP_DIR, str(video_id))
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError as exc:
            logger.warning("embedding_tmp_cleanup_failed", video_id=str(video_id), reason=str(exc))
