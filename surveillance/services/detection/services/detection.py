"""Orchestrator. process_frames() is the entry point every message reaches.

Fixed against the real shared.events.detection_complete schema:
- Detection/DetectionCompleteEvent are now built using the actual flat
  shape (detections: list[Detection], nested BoundingBox, frame_count,
  video_id as str) instead of an invented FrameDetections/DetectionMetadata
  shape that doesn't exist in shared/.
- scene_id/timestamp_ms/sequence_index — which the real Detection model has
  no dedicated field for — are carried in Detection.extra so embedding can
  still deep-link a crop to a video timestamp without an extra DB lookup.
- DetectionResult rows are cleared for the video_id before re-inserting,
  closing the duplicate-row-on-redelivery gap (MinIO overwrites are
  idempotent; SQL INSERTs are not).
- Idempotency uses VideoStatus.DETECTED (added to shared/models/video.py).
"""
from __future__ import annotations

import os
import shutil
import uuid

from sqlalchemy import delete

from shared.events.detection_complete import (
    BoundingBox,
    Detection,
    DetectionCompleteEvent,
)
from shared.events.frame_extracted import FramesExtractedEvent
from shared.models.detection_result import DetectionResult
from shared.models.video import VideoRecord, VideoStatus

from services.detection.core.config import Settings
from services.detection.core.logging import get_logger
from services.detection.db.database import get_session
from services.detection.models.schemas import (
    DetectionError,
    DetectionStage,
    FrameRef,
    PersistedDetection,
    PersistedFrameResult,
)
from services.detection.services.grounding_dino import GroundingDINODetector
from services.detection.services.queue import DetectionPublisher
from services.detection.services.storage import DetectionStorageService

logger = get_logger(__name__)


class DetectionService:
    def __init__(
        self,
        settings: Settings,
        detector: GroundingDINODetector,
        storage: DetectionStorageService,
        publisher: DetectionPublisher,
    ):
        self._settings = settings
        self._detector = detector
        self._storage = storage
        self._publisher = publisher

    async def process_frames(self, event: FramesExtractedEvent) -> None:
        video_id = event.video_id

        if await self._already_processed(video_id):
            logger.info("detection_already_processed_skipped", video_id=str(video_id))
            return

        await self._mark_status(video_id, VideoStatus.PROCESSING)
        # Clear any partial rows from a prior crashed/redelivered attempt
        # before writing new ones — closes the duplicate-row gap, since
        # unlike MinIO put_object, a plain INSERT is not idempotent on retry.
        await self._clear_existing_results(video_id)

        try:
            frame_results = await self._run_detection(video_id, event)
            stored_event = self._build_event(video_id, frame_results)
            await self._publisher.publish_detection_complete(stored_event)
            await self._mark_status(video_id, VideoStatus.DETECTED)
        except DetectionError as exc:
            if exc.recoverable:
                logger.warning(
                    "detection_recoverable_failure",
                    video_id=str(video_id),
                    stage=exc.stage.value,
                    reason=exc.message,
                )
                raise
            await self._quarantine_and_fail(video_id, event, exc.message)
        finally:
            self._cleanup_tmp_files(video_id)

    async def _run_detection(
        self, video_id: uuid.UUID, event: FramesExtractedEvent
    ) -> list[PersistedFrameResult]:
        results: list[PersistedFrameResult] = []

        for frame in event.frames:
            ref = FrameRef(
                frame_path=frame.frame_path,
                sequence_index=frame.sequence_index,
                timestamp_ms=frame.timestamp_ms,
                scene_id=frame.scene_id,
            )

            local_path = await self._fetch_and_write_frame(video_id, ref)

            raw_detections = await self._detector.detect(local_path)
            kept = [
                d
                for d in raw_detections
                if d.confidence >= self._settings.DETECTION_CONFIDENCE_THRESHOLD
            ]

            persisted = await self._persist_detections(video_id, ref, local_path, kept)
            results.append(PersistedFrameResult(frame=ref, detections=persisted))

        return results

    async def _fetch_and_write_frame(self, video_id: uuid.UUID, ref: FrameRef) -> str:
        try:
            data = await self._storage.fetch_frame(
                self._settings.MINIO_PROCESSED_FRAMES_BUCKET, ref.frame_path
            )
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(
                message=f"Failed to fetch frame {ref.frame_path}: {exc}",
                stage=DetectionStage.FETCH_FRAMES,
                recoverable=True,
            ) from exc

        tmp_dir = os.path.join(self._settings.DETECTION_TMP_DIR, str(video_id))
        os.makedirs(tmp_dir, exist_ok=True)
        local_path = os.path.join(tmp_dir, f"{ref.sequence_index:06d}.jpg")
        with open(local_path, "wb") as f:
            f.write(data)
        return local_path

    async def _persist_detections(
        self,
        video_id: uuid.UUID,
        ref: FrameRef,
        local_frame_path: str,
        raw_detections: list,
    ) -> list[PersistedDetection]:
        persisted: list[PersistedDetection] = []
        if not raw_detections:
            return persisted

        from PIL import Image

        image = Image.open(local_frame_path).convert("RGB")
        width, height = image.size

        rows: list[DetectionResult] = []
        for raw in raw_detections:
            detection_id = uuid.uuid4()
            crop_path: str | None = None
            crop_failed = False

            try:
                box_px = (
                    int(raw.bbox_x1 * width),
                    int(raw.bbox_y1 * height),
                    int(raw.bbox_x2 * width),
                    int(raw.bbox_y2 * height),
                )
                crop = image.crop(box_px)
                import io

                buf = io.BytesIO()
                crop.save(buf, format="JPEG", quality=90)
                _, crop_path = await self._storage.upload_crop(
                    video_id, ref.sequence_index, detection_id, buf.getvalue()
                )
            except Exception as exc:  # noqa: BLE001
                crop_failed = True
                logger.warning(
                    "detection_crop_persist_failed",
                    video_id=str(video_id),
                    sequence_index=ref.sequence_index,
                    reason=str(exc),
                )

            persisted.append(
                PersistedDetection(detection_id=detection_id, raw=raw, crop_path=crop_path)
            )
            rows.append(
                DetectionResult(
                    id=detection_id,
                    video_id=video_id,
                    frame_path=ref.frame_path,
                    frame_timestamp_ms=ref.timestamp_ms,
                    scene_id=ref.scene_id,
                    label=raw.label,
                    confidence=raw.confidence,
                    bbox_x1=raw.bbox_x1,
                    bbox_y1=raw.bbox_y1,
                    bbox_x2=raw.bbox_x2,
                    bbox_y2=raw.bbox_y2,
                    crop_path=crop_path,
                )
            )
            # crop_failed is intentionally not persisted as a DB column (no
            # such column exists) — it's surfaced downstream instead, via
            # Detection.extra["crop_status"] in _build_event, so embedding
            # can distinguish "no crop by design" from "crop upload failed".
            persisted[-1].raw.__dict__.setdefault("_crop_failed", crop_failed)  # type: ignore[attr-defined]

        await self._write_detection_rows(rows)
        return persisted

    async def _write_detection_rows(self, rows: list[DetectionResult]) -> None:
        if not rows:
            return
        async with get_session() as db:
            for row in rows:
                db.add(row)

    async def _clear_existing_results(self, video_id: uuid.UUID) -> None:
        async with get_session() as db:
            await db.execute(delete(DetectionResult).where(DetectionResult.video_id == video_id))

    def _build_event(
        self, video_id: uuid.UUID, frame_results: list[PersistedFrameResult]
    ) -> DetectionCompleteEvent:
        """Conforms to the real shared.events.detection_complete schema:
        a flat list of Detection, each with a nested BoundingBox. scene_id/
        timestamp_ms/sequence_index ride in `extra` since the real schema
        has no dedicated fields for them.
        """
        detections_payload: list[Detection] = []
        for result in frame_results:
            for d in result.detections:
                extra: dict = {
                    "scene_id": result.frame.scene_id,
                    "timestamp_ms": result.frame.timestamp_ms,
                    "sequence_index": result.frame.sequence_index,
                }
                if getattr(d.raw, "_crop_failed", False):
                    extra["crop_status"] = "failed"
                detections_payload.append(
                    Detection(
                        frame_path=result.frame.frame_path,
                        label=d.raw.label,
                        confidence=d.raw.confidence,
                        bbox=BoundingBox(
                            x1=d.raw.bbox_x1,
                            y1=d.raw.bbox_y1,
                            x2=d.raw.bbox_x2,
                            y2=d.raw.bbox_y2,
                        ),
                        crop_path=d.crop_path,
                        extra=extra,
                    )
                )

        return DetectionCompleteEvent(
            video_id=str(video_id),
            detections=detections_payload,
            frame_count=len(frame_results),
        )

    async def _already_processed(self, video_id: uuid.UUID) -> bool:
        async with get_session() as db:
            record = await db.get(VideoRecord, video_id)
            if record is None:
                logger.warning("detection_video_record_missing", video_id=str(video_id))
                return False
            return record.status in (VideoStatus.DETECTED, VideoStatus.INDEXED)

    async def _mark_status(
        self, video_id: uuid.UUID, status: VideoStatus, error_message: str | None = None
    ) -> None:
        async with get_session() as db:
            record = await db.get(VideoRecord, video_id)
            if record is None:
                logger.warning(
                    "detection_video_record_missing_on_status_update",
                    video_id=str(video_id),
                    status=status.value,
                )
                return
            record.status = status
            if error_message is not None:
                record.error_message = error_message

    async def _quarantine_and_fail(
        self, video_id: uuid.UUID, event: FramesExtractedEvent, reason: str
    ) -> None:
        try:
            if event.frames:
                first_frame = event.frames[0]
                data = await self._storage.fetch_frame(
                    self._settings.MINIO_PROCESSED_FRAMES_BUCKET, first_frame.frame_path
                )
                await self._storage.quarantine(video_id, first_frame.frame_path, data)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "detection_quarantine_upload_failed", video_id=str(video_id), reason=str(exc)
            )
        await self._mark_status(video_id, VideoStatus.FAILED, error_message=reason)

    def _cleanup_tmp_files(self, video_id: uuid.UUID) -> None:
        tmp_dir = os.path.join(self._settings.DETECTION_TMP_DIR, str(video_id))
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError as exc:
            logger.warning("detection_tmp_cleanup_failed", video_id=str(video_id), reason=str(exc))