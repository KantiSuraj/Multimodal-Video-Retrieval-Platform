"""
services/preprocessing/services/preprocessing.py

The orchestrator — the preprocessing equivalent of
services/ingestion/services/ingestion.py's IngestionService. This is the
single most important file in the service: PreprocessingService.process_video()
is the eight-stage sequence every ingested video runs through, exactly the
way _run_pipeline() is the six-step sequence every upload runs through.

Failure philosophy, mirrored from ingestion:
  - Transient infra problems (MinIO down, FFmpeg subprocess timeout) are
    `recoverable=True`: this method re-raises, the queue consumer's
    `message.process()` nacks, RabbitMQ redelivers later. No DB write is
    left half-done because nothing is committed until the pipeline
    finishes (mirrors ingestion's flush-before-commit-after-handler shape,
    adapted to a worker with no HTTP request boundary).
  - Bad/corrupt input (FFmpeg can't decode it, zero frames survive
    quality filtering) is `recoverable=False`: this method quarantines the
    source video's bytes (audit trail, same as ingestion's
    _quarantine()), marks VideoRecord.status = FAILED with an
    error_message, and does NOT re-raise — the message acks so RabbitMQ
    does not redeliver a video that will never succeed.
  - Idempotency: VideoRecord.status is checked before any work begins. If
    a video is already PREPROCESSED or further along, this method returns
    immediately. This handles RabbitMQ's at-least-once delivery the same
    way ingestion's SHA-256 dedup check handles duplicate uploads.

This service NEVER performs detection, embedding, or indexing. Its output
ends at FramesExtractedEvent.
"""
from __future__ import annotations

import os
import shutil
import uuid

from sqlalchemy import select

from shared.shared.events.frame_extracted import (
    ExtractedFrame,
    FramesExtractedEvent,
    GeneratedClip,
    PreprocessingMetadata,
    SceneBoundary,
)
from shared.shared.events.video_ingested import VideoIngestedEvent
from shared.shared.models.video import VideoRecord, VideoStatus

from services.preprocessing.core.config import Settings
from services.preprocessing.core.logging import get_logger
from services.preprocessing.db.database import get_session
from services.preprocessing.models.schemas import PreprocessingError, PreprocessingResult, PreprocessingStage
from services.preprocessing.services.clip_generator import ClipGenerator
from services.preprocessing.services.enhancer import FrameEnhancer
from services.preprocessing.services.frame_extractor import FrameExtractor
from services.preprocessing.services.quality_filter import QualityFilter
from services.preprocessing.services.queue import PreprocessingPublisher
from services.preprocessing.services.scene_segmenter import SceneSegmenter
from services.preprocessing.services.storage import PreprocessingStorageService
from services.preprocessing.services.transcoder import VideoTranscoder

logger = get_logger(__name__)


class PreprocessingService:
    """Stateless, like IngestionService — safe to share a single instance
    across all messages a consumer processes."""

    def __init__(
        self,
        settings: Settings,
        storage_service: PreprocessingStorageService,
        publisher: PreprocessingPublisher,
    ) -> None:
        self._settings = settings
        self._storage = storage_service
        self._publisher = publisher
        self._transcoder = VideoTranscoder(settings)
        self._frame_extractor = FrameExtractor(settings)
        self._quality_filter = QualityFilter(settings)
        self._enhancer = FrameEnhancer(settings)
        self._scene_segmenter = SceneSegmenter(settings)
        self._clip_generator = ClipGenerator(settings)

    async def process_video(self, event: VideoIngestedEvent) -> None:
        video_id = event.video_id
        raw_bytes: bytes | None = None

        try:
            if not await self._claim_for_processing(video_id):
                logger.info("preprocessing_skip_already_done_or_in_flight", video_id=str(video_id))
                return

            await self._mark_status(video_id, VideoStatus.PROCESSING)

            raw_bytes = await self._fetch_source(event)
            result = await self._run_stages(video_id, raw_bytes, event)
            stored_event = await self._persist_artifacts(video_id, result)

            await self._publisher.publish_frames_extracted(stored_event)
            await self._mark_status(video_id, VideoStatus.PREPROCESSED)

            logger.info(
                "preprocessing_complete",
                video_id=str(video_id),
                frame_count=len(result.frames),
                scene_count=len(result.scenes),
                clip_count=len(result.clips),
            )

        except PreprocessingError as exc:
            if exc.recoverable:
                logger.error(
                    "preprocessing_recoverable_failure",
                    video_id=str(video_id),
                    stage=exc.stage,
                    error=exc.message,
                )
                raise  # nack + requeue
            await self._quarantine_and_fail(video_id, event, raw_bytes_fallback=raw_bytes, reason=exc.message)
        finally:
            self._cleanup_tmp_files(video_id)

    def _cleanup_tmp_files(self, video_id: uuid.UUID) -> None:
        """Best-effort cleanup of every local intermediate this video_id may
        have produced across stages. Failure to clean up is logged, never
        raised — a leftover temp file should never fail the pipeline."""
        candidates = [
            os.path.join(self._settings.TMP_DIR, f"{video_id}_source"),
            os.path.join(self._settings.TMP_DIR, f"{video_id}_normalized.mp4"),
            os.path.join(self._settings.TMP_DIR, f"{video_id}_frames"),
            os.path.join(self._settings.TMP_DIR, f"{video_id}_enhanced"),
            os.path.join(self._settings.TMP_DIR, f"{video_id}_clips"),
        ]
        for path in candidates:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                logger.warning("tmp_cleanup_failed", path=path, error=str(exc))

    async def _run_stages(
        self, video_id: uuid.UUID, raw_bytes: bytes, event: VideoIngestedEvent
    ) -> PreprocessingResult:
        os.makedirs(self._settings.TMP_DIR, exist_ok=True)
        source_path = os.path.join(self._settings.TMP_DIR, f"{video_id}_source")
        with open(source_path, "wb") as f:
            f.write(raw_bytes)

        # Stage 1 — normalization
        normalized = await self._transcoder.normalize(video_id, source_path)
        video_duration_ms = int(normalized.duration_seconds * 1000)

        # Stage 2 — frame extraction
        candidates = await self._frame_extractor.extract(video_id, normalized.local_path)

        # Stage 3 — quality filtering
        sharpness_by_index = {
            c.sequence_index: (self._quality_filter.score(c.local_path) or 0.0) for c in candidates
        }
        quality_result = self._quality_filter.filter(candidates)

        if not quality_result.kept:
            raise PreprocessingError(
                "All extracted frames rejected by quality filter — "
                "video may be uniformly out-of-focus or corrupt",
                stage=PreprocessingStage.QUALITY_FILTER,
                recoverable=False,
            )

        # Stage 4 — CLAHE enhancement
        enhanced_frames = self._enhancer.enhance_batch(
            quality_result.kept, sharpness_by_index, video_id
        )

        # Stage 5 — scene segmentation
        scenes = self._scene_segmenter.segment(enhanced_frames, video_duration_ms)

        # Stage 6 — clip generation
        clips = await self._clip_generator.generate(
            video_id, normalized.local_path, scenes, video_duration_ms
        )

        return PreprocessingResult(
            frames=enhanced_frames,
            scenes=scenes,
            clips=clips,
            normalized_video_local_path=normalized.local_path,
            normalized=normalized,
        )
    async def _persist_artifacts(
            self, video_id: uuid.UUID, result: PreprocessingResult
        ) -> FramesExtractedEvent:
            uploaded: list[tuple[str, str]] = []  # (bucket, key) for rollback

            try:
                with open(result.normalized_video_local_path, "rb") as f:
                    video_bytes = f.read()
                processed_video_path = await self._storage.upload_processed_video(video_id, video_bytes)
                uploaded.append((self._settings.MINIO_PROCESSED_VIDEO_BUCKET, processed_video_path))

                extracted_frames: list[ExtractedFrame] = []
                for frame in result.frames:
                    with open(frame.local_path, "rb") as f:
                        frame_bytes = f.read()
                    frame_path = await self._storage.upload_frame(video_id, frame.sequence_index, frame_bytes)
                    uploaded.append((self._settings.MINIO_PROCESSED_FRAMES_BUCKET, frame_path))
                    scene_id = self._scene_segmenter.scene_for_timestamp(result.scenes, frame.timestamp_ms)
                    extracted_frames.append(
                        ExtractedFrame(
                            frame_path=frame_path,
                            sequence_index=frame.sequence_index,
                            timestamp_ms=frame.timestamp_ms,
                            scene_id=scene_id,
                            sharpness_score=frame.sharpness_score,
                        )
                    )

                generated_clips: list[GeneratedClip] = []
                for clip in result.clips:
                    with open(clip.local_path, "rb") as f:
                        clip_bytes = f.read()
                    clip_path = await self._storage.upload_clip(video_id, clip.scene_id, clip_bytes)
                    uploaded.append((self._settings.MINIO_PROCESSED_CLIPS_BUCKET, clip_path))
                    generated_clips.append(
                        GeneratedClip(
                            clip_path=clip_path,
                            scene_id=clip.scene_id,
                            start_ms=clip.start_ms,
                            end_ms=clip.end_ms,
                            duration_seconds=clip.duration_seconds,
                        )
                    )
            except Exception:
                logger.warning("persistence_failed_rolling_back", video_id=str(video_id), uploaded_count=len(uploaded))
                for bucket, key in uploaded:
                    await self._storage.delete_artifact(bucket, key)
                raise PreprocessingError(
                    "Failed to persist artifacts to MinIO",
                    stage=PreprocessingStage.PERSISTENCE,
                    recoverable=True,
                )

            scene_boundaries = [
                SceneBoundary(scene_id=s.scene_id, start_ms=s.start_ms, end_ms=s.end_ms)
                for s in result.scenes
            ]

            return FramesExtractedEvent(
                video_id=video_id,
                processed_video_path=processed_video_path,
                processed_video_bucket=self._settings.MINIO_PROCESSED_VIDEO_BUCKET,
                frames=extracted_frames,
                scenes=scene_boundaries,
                clips=generated_clips,
                preprocessing_metadata=PreprocessingMetadata(
                    extraction_interval_seconds=self._settings.FRAME_EXTRACTION_INTERVAL_SECONDS,
                    blur_threshold=self._settings.BLUR_LAPLACIAN_VARIANCE_THRESHOLD,
                    clahe_clip_limit=self._settings.CLAHE_CLIP_LIMIT,
                    clahe_tile_grid_size=self._settings.CLAHE_TILE_GRID_SIZE,
                    scene_histogram_threshold=self._settings.SCENE_HISTOGRAM_DIFF_THRESHOLD,
                    target_clip_duration_seconds=self._settings.DEFAULT_CLIP_DURATION_SECONDS,
                    normalized_codec=self._settings.NORMALIZED_CODEC,
                    normalized_resolution=self._settings.NORMALIZED_RESOLUTION,
                    normalized_fps=self._settings.NORMALIZED_FPS,
                ),
            )

    # @staticmethod
    # def _scene_for_timestamp(scenes, timestamp_ms: int) -> int:
    #     for scene in scenes:
    #         if scene.start_ms <= timestamp_ms < scene.end_ms:
    #             return scene.scene_id
    #     return scenes[-1].scene_id if scenes else 0

    async def _fetch_source(self, event: VideoIngestedEvent) -> bytes:
        try:
            return await self._storage.fetch_raw_video(event.storage_bucket, event.storage_path)
        except Exception as exc:  # MinIO already retried 3x internally
            raise PreprocessingError(
                f"Failed to fetch source video from MinIO: {exc}",
                stage=PreprocessingStage.FETCH_SOURCE,
                recoverable=True,
            ) from exc

    async def _claim_for_processing(self, video_id: uuid.UUID) -> bool:
        """Atomically checks status and claims PROCESSING under a row lock,
        closing the race window between the idempotency check and the
        status write."""
        async with get_session() as db:
            record = (
                await db.execute(
                    select(VideoRecord).where(VideoRecord.id == video_id).with_for_update()
                )
            ).scalar_one_or_none()
            if record is None:
                logger.warning("video_record_missing", video_id=str(video_id))
                return False
            if record.status in (VideoStatus.PREPROCESSED, VideoStatus.INDEXED, VideoStatus.PROCESSING):
                return False
            record.status = VideoStatus.PROCESSING
            return True 
        

    async def _mark_status(
        self, video_id: uuid.UUID, status: VideoStatus, error_message: str | None = None
    ) -> None:
        async with get_session() as db:
            record = await db.get(VideoRecord, video_id)
            if record is None:
                logger.warning("video_record_missing", video_id=str(video_id))
                return
            record.status = status
            record.error_message = error_message

    async def _quarantine_and_fail(
        self,
        video_id: uuid.UUID,
        event: VideoIngestedEvent,
        raw_bytes_fallback: bytes | None,
        reason: str,
    ) -> None:
        try:
            raw_bytes = raw_bytes_fallback
            if raw_bytes is None:
                raw_bytes = await self._storage.fetch_raw_video(
                    event.storage_bucket, event.storage_path
                )
            await self._storage.quarantine(video_id, event.original_filename, raw_bytes)
        except Exception as exc:  # quarantine itself failing is non-fatal, same as ingestion
            logger.error("quarantine_upload_failed", video_id=str(video_id), error=str(exc))

        await self._mark_status(video_id, VideoStatus.FAILED, error_message=reason)
        logger.error("preprocessing_quarantined", video_id=str(video_id), reason=reason)
