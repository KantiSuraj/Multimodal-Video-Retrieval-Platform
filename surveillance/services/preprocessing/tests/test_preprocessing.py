from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest

from services.preprocessing.models.schemas import (
    EnhancedFrame,
    FrameCandidate,
    PreprocessingError,
    PreprocessingStage,
    SceneSegment,
)
from services.preprocessing.services.clip_generator import ClipGenerator
from services.preprocessing.services.preprocessing import PreprocessingService
from services.preprocessing.services.quality_filter import QualityFilter
from services.preprocessing.services.scene_segmenter import SceneSegmenter
from shared.models.video import VideoRecord, VideoStatus


def _write_image(path: str, sharp: bool, color: tuple[int, int, int] = (120, 120, 120)) -> None:
    """Writes a synthetic test image. `sharp=True` draws high-frequency
    noise/edges (high Laplacian variance); `sharp=False` writes a flat,
    uniform image (near-zero Laplacian variance) to simulate blur."""
    img = np.full((64, 64, 3), color, dtype=np.uint8)
    if sharp:
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        img = noise
        cv2.rectangle(img, (5, 5), (58, 58), (0, 0, 0), 2)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------------------
# TestQualityFilter
# ---------------------------------------------------------------------------


class TestQualityFilter:
    def test_sharp_frame_kept(self, settings, tmp_path):
        _write_image(str(tmp_path / "sharp.jpg"), sharp=True)
        qf = QualityFilter(settings)
        candidate = FrameCandidate(0, 0, str(tmp_path / "sharp.jpg"))

        result = qf.filter([candidate])

        assert result.kept == [candidate]
        assert result.rejected_count == 0

    def test_blurry_frame_rejected(self, settings, tmp_path):
        _write_image(str(tmp_path / "blurry.jpg"), sharp=False)
        qf = QualityFilter(settings)
        candidate = FrameCandidate(0, 0, str(tmp_path / "blurry.jpg"))

        result = qf.filter([candidate])

        assert result.kept == []
        assert result.rejected_count == 1

    def test_unreadable_frame_rejected_not_raised(self, settings):
        qf = QualityFilter(settings)
        candidate = FrameCandidate(0, 0, "/nonexistent/path.jpg")

        result = qf.filter([candidate])

        assert result.kept == []
        assert result.rejected_count == 1

    def test_threshold_is_configurable(self, settings, tmp_path):
        _write_image(str(tmp_path / "mid.jpg"), sharp=True)
        qf = QualityFilter(settings)
        score = qf.score(str(tmp_path / "mid.jpg"))

        settings.BLUR_LAPLACIAN_VARIANCE_THRESHOLD = score + 1_000_000
        result = qf.filter([FrameCandidate(0, 0, str(tmp_path / "mid.jpg"))])
        assert result.kept == []


# ---------------------------------------------------------------------------
# TestSceneSegmenter
# ---------------------------------------------------------------------------


class TestSceneSegmenter:
    def test_uniform_video_yields_single_scene(self, settings, tmp_path):
        frames = []
        for i in range(5):
            path = str(tmp_path / f"f{i}.jpg")
            _write_image(path, sharp=False, color=(100, 100, 100))
            frames.append(EnhancedFrame(i, i * 1000, path, sharpness_score=1.0))

        segmenter = SceneSegmenter(settings)
        scenes = segmenter.segment(frames, video_duration_ms=5000)

        assert len(scenes) == 1
        assert scenes[0].start_ms == 0
        assert scenes[0].end_ms == 5000

    def test_color_change_creates_boundary(self, settings, tmp_path):
        frames = []
        for i in range(3):
            path = str(tmp_path / f"a{i}.jpg")
            _write_image(path, sharp=False, color=(20, 20, 200))  # red-ish
            frames.append(EnhancedFrame(i, i * 1000, path, sharpness_score=1.0))
        for i in range(3, 6):
            path = str(tmp_path / f"b{i}.jpg")
            _write_image(path, sharp=False, color=(200, 20, 20))  # blue-ish
            frames.append(EnhancedFrame(i, i * 1000, path, sharpness_score=1.0))

        settings.SCENE_HISTOGRAM_DIFF_THRESHOLD = 0.1
        segmenter = SceneSegmenter(settings)
        scenes = segmenter.segment(frames, video_duration_ms=6000)

        assert len(scenes) >= 2

    def test_empty_frame_list_yields_no_scenes(self, settings):
        segmenter = SceneSegmenter(settings)
        assert segmenter.segment([], video_duration_ms=0) == []


# ---------------------------------------------------------------------------
# TestClipGenerator (windowing logic only — no real FFmpeg subprocess)
# ---------------------------------------------------------------------------


class TestClipGenerator:
    def test_short_scene_extended_to_minimum(self, settings):
        gen = ClipGenerator(settings)
        scene = SceneSegment(scene_id=0, start_ms=10_000, end_ms=10_500)  # 0.5s, below 2s min

        windows = gen._clip_windows(scene, video_duration_ms=60_000)

        assert len(windows) == 1
        start, end = windows[0]
        assert (end - start) >= int(settings.MIN_CLIP_DURATION_SECONDS * 1000)

    def test_normal_scene_kept_as_single_window(self, settings):
        gen = ClipGenerator(settings)
        scene = SceneSegment(scene_id=0, start_ms=0, end_ms=4000)  # 4s, within [2,30]

        windows = gen._clip_windows(scene, video_duration_ms=60_000)

        assert windows == [(0, 4000)]

    def test_long_scene_split_into_max_duration_windows(self, settings):
        gen = ClipGenerator(settings)
        scene = SceneSegment(scene_id=0, start_ms=0, end_ms=70_000)  # 70s, above 30s max

        windows = gen._clip_windows(scene, video_duration_ms=70_000)

        assert len(windows) == 3  # 30 + 30 + 10
        for start, end in windows:
            assert (end - start) <= int(settings.MAX_CLIP_DURATION_SECONDS * 1000)

    def test_short_scene_at_video_start_does_not_go_negative(self, settings):
        gen = ClipGenerator(settings)
        scene = SceneSegment(scene_id=0, start_ms=0, end_ms=200)

        windows = gen._clip_windows(scene, video_duration_ms=60_000)

        start, end = windows[0]
        assert start >= 0


# ---------------------------------------------------------------------------
# TestPreprocessingService (orchestrator — failure modes & idempotency)
# ---------------------------------------------------------------------------


def _make_service(settings, monkeypatch) -> tuple[PreprocessingService, AsyncMock, AsyncMock]:
    storage = AsyncMock()
    publisher = AsyncMock()
    service = PreprocessingService(settings, storage, publisher)
    return service, storage, publisher


class TestPreprocessingServiceIdempotency:
    @pytest.mark.asyncio
    async def test_already_preprocessed_video_is_skipped(
        self, settings, video_ingested_event, monkeypatch
    ):
        service, storage, publisher = _make_service(settings, monkeypatch)
        record = MagicMock(spec=VideoRecord)
        record.status = VideoStatus.PREPROCESSED

        session_mock = AsyncMock()
        session_mock.get = AsyncMock(return_value=record)

        monkeypatch.setattr(
            "services.preprocessing.services.preprocessing.get_session",
            lambda: _FakeSessionCtx(session_mock),
        )

        await service.process_video(video_ingested_event)

        storage.fetch_raw_video.assert_not_called()
        publisher.publish_frames_extracted.assert_not_called()


class _FakeSessionCtx:
    """Minimal async context manager standing in for get_session()."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


class TestPreprocessingServiceFailureModes:
    @pytest.mark.asyncio
    async def test_recoverable_fetch_failure_propagates_for_requeue(
        self, settings, video_ingested_event, monkeypatch
    ):
        service, storage, publisher = _make_service(settings, monkeypatch)
        storage.fetch_raw_video.side_effect = ConnectionError("minio down")

        session_mock = AsyncMock()
        session_mock.get = AsyncMock(return_value=MagicMock(status=VideoStatus.PENDING))
        monkeypatch.setattr(
            "services.preprocessing.services.preprocessing.get_session",
            lambda: _FakeSessionCtx(session_mock),
        )

        with pytest.raises(PreprocessingError) as exc_info:
            await service.process_video(video_ingested_event)

        assert exc_info.value.recoverable is True
        assert exc_info.value.stage == PreprocessingStage.FETCH_SOURCE

    @pytest.mark.asyncio
    async def test_non_recoverable_failure_quarantines_and_does_not_raise(
        self, settings, video_ingested_event, monkeypatch
    ):
        service, storage, publisher = _make_service(settings, monkeypatch)
        storage.fetch_raw_video.return_value = b"not a real video"

        session_mock = AsyncMock()
        session_mock.get = AsyncMock(return_value=MagicMock(status=VideoStatus.PENDING))
        monkeypatch.setattr(
            "services.preprocessing.services.preprocessing.get_session",
            lambda: _FakeSessionCtx(session_mock),
        )

        async def fake_run_stages(video_id, raw_bytes, event):
            raise PreprocessingError(
                "corrupt video", stage=PreprocessingStage.NORMALIZATION, recoverable=False
            )

        monkeypatch.setattr(service, "_run_stages", fake_run_stages)

        # Should not raise — message acks, video is quarantined instead.
        await service.process_video(video_ingested_event)

        storage.quarantine.assert_awaited_once()
        publisher.publish_frames_extracted.assert_not_called()
