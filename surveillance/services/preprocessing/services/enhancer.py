"""
services/preprocessing/services/enhancer.py — Stage 4: CLAHE enhancement

CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes local
contrast per-tile rather than globally, which matters for surveillance
footage specifically: a single frame often has both a bright window and a
dark hallway in it, and a global histogram equalization would blow out one
to fix the other. CLAHE fixes each tile independently and clips the
contrast amplification (`clip_limit`) so noise in flat regions isn't
amplified into visible artifacts.

Applied on the L channel of LAB color space, not directly on BGR — this
preserves color information (a/b channels) while only adjusting lightness,
which avoids the color-shift artifacts you get from equalizing each BGR
channel independently.

A frame that fails to load or fails enhancement is dropped, mirroring
quality_filter.py's per-frame failure handling — one bad frame does not
fail the batch.
"""
from __future__ import annotations

import os
import uuid

import cv2

from services.preprocessing.core.config import Settings
from services.preprocessing.core.logging import get_logger
from services.preprocessing.models.schemas import EnhancedFrame, FrameCandidate

logger = get_logger(__name__)


class FrameEnhancer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._clahe = cv2.createCLAHE(
            clipLimit=settings.CLAHE_CLIP_LIMIT,
            tileGridSize=(settings.CLAHE_TILE_GRID_SIZE, settings.CLAHE_TILE_GRID_SIZE),
        )

    def enhance_one(self, candidate: FrameCandidate, video_id: uuid.UUID, sharpness_score: float) -> EnhancedFrame | None:
        image = cv2.imread(candidate.local_path, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("enhancement_frame_unreadable", path=candidate.local_path)
            return None

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_channel)
        merged = cv2.merge((l_enhanced, a_channel, b_channel))
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        enhanced_dir = os.path.join(self._settings.TMP_DIR, f"{video_id}_enhanced")
        os.makedirs(enhanced_dir, exist_ok=True)
        out_path = os.path.join(enhanced_dir, f"{candidate.sequence_index:06d}.jpg")
        cv2.imwrite(out_path, result)

        return EnhancedFrame(
            sequence_index=candidate.sequence_index,
            timestamp_ms=candidate.timestamp_ms,
            local_path=out_path,
            sharpness_score=sharpness_score,
        )

    def enhance_batch(
        self,
        candidates: list[FrameCandidate],
        sharpness_by_index: dict[int, float],
        video_id: uuid.UUID,
    ) -> list[EnhancedFrame]:
        enhanced: list[EnhancedFrame] = []
        for candidate in candidates:
            result = self.enhance_one(
                candidate, video_id, sharpness_by_index.get(candidate.sequence_index, 0.0)
            )
            if result is not None:
                enhanced.append(result)
        return enhanced
