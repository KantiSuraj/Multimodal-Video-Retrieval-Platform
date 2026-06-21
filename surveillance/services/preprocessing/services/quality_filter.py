"""
services/preprocessing/services/quality_filter.py — Stage 3: quality filtering

Laplacian variance is the standard, cheap blur metric: convolve with the
Laplacian kernel and take the variance of the response. A sharp image has
strong edges everywhere -> high variance. A blurry image has smoothed-out
edges -> low variance. Below `BLUR_LAPLACIAN_VARIANCE_THRESHOLD`, a frame
is rejected before it ever reaches CLAHE or downstream detection — there's
no point enhancing or shipping a frame nothing useful can be detected in.

A single corrupt/unreadable frame is treated as a per-frame failure, not a
pipeline failure (one bad JPEG from FFmpeg's output shouldn't kill the
whole video) — it's logged and excluded, mirroring the spirit of
ingestion's "validation failure quarantines the file, doesn't crash the
service" philosophy at the unit-of-work level.
"""
from __future__ import annotations

import cv2

from services.preprocessing.core.config import Settings
from services.preprocessing.core.logging import get_logger
from services.preprocessing.models.schemas import FrameCandidate, QualityFilterResult

logger = get_logger(__name__)


class QualityFilter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def score(self, local_path: str) -> float | None:
        image = cv2.imread(local_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return float(cv2.Laplacian(image, cv2.CV_64F).var())

    def filter(self, candidates: list[FrameCandidate]) -> QualityFilterResult:
        kept: list[FrameCandidate] = []
        rejected_count = 0

        for candidate in candidates:
            sharpness = self.score(candidate.local_path)
            if sharpness is None:
                logger.warning("frame_unreadable", path=candidate.local_path)
                rejected_count += 1
                continue
            if sharpness < self._settings.BLUR_LAPLACIAN_VARIANCE_THRESHOLD:
                rejected_count += 1
                continue
            kept.append(candidate)

        return QualityFilterResult(kept=kept, rejected_count=rejected_count)
