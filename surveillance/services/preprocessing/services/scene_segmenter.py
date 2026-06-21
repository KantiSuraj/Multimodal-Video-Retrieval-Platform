"""
services/preprocessing/services/scene_segmenter.py — Stage 5: scene segmentation

Computes a color histogram (HSV, hue+saturation channels) for each
enhanced frame and compares consecutive frames with correlation distance.
When the difference exceeds `SCENE_HISTOGRAM_DIFF_THRESHOLD`, a new scene
boundary is declared. This is the standard cheap approach to shot-boundary
detection — no ML model needed, which matters because preprocessing must
not own anything resembling detection/embedding (see architectural
boundary notes). It operates purely on frame statistics.

Scene boundaries are expressed in timestamp_ms, derived from the enhanced
frames' own timestamps (already fixed by the extraction interval), so this
stage has no dependency on frame-rate assumptions beyond what extraction
already produced.
"""
from __future__ import annotations

import cv2
import numpy as np

from services.preprocessing.core.config import Settings
from services.preprocessing.core.logging import get_logger
from services.preprocessing.models.schemas import EnhancedFrame, SceneSegment

logger = get_logger(__name__)


class SceneSegmenter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _histogram(self, local_path: str) -> np.ndarray | None:
        image = cv2.imread(local_path, cv2.IMREAD_COLOR)
        if image is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist
    
    @staticmethod
    def scene_for_timestamp(scenes: list[SceneSegment], timestamp_ms: int) -> int:
        for scene in scenes:
            if scene.start_ms <= timestamp_ms < scene.end_ms:
                return scene.scene_id
        return scenes[-1].scene_id if scenes else 0

    def segment(self, frames: list[EnhancedFrame], video_duration_ms: int) -> list[SceneSegment]:
        if not frames:
            return []

        boundaries: list[int] = [0]
        prev_hist = self._histogram(frames[0].local_path)

        for frame in frames[1:]:
            curr_hist = self._histogram(frame.local_path)
            if prev_hist is None or curr_hist is None:
                prev_hist = curr_hist
                continue

            correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            diff = 1.0 - correlation  # 0 = identical, 2 = maximally different

            if diff >= self._settings.SCENE_HISTOGRAM_DIFF_THRESHOLD:
                boundaries.append(frame.timestamp_ms)

            prev_hist = curr_hist

        boundaries.append(video_duration_ms)
        boundaries = sorted(set(boundaries))

        scenes = [
            SceneSegment(scene_id=i, start_ms=start, end_ms=end)
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
            if end > start
        ]
        return scenes or [SceneSegment(scene_id=0, start_ms=0, end_ms=video_duration_ms)]
