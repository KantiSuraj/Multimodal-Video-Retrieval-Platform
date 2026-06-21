"""
services/preprocessing/services/frame_extractor.py — Stage 2: frame extraction

Extracts frames from the normalized video at a configurable interval
(default 1 frame/sec). Uses FFmpeg's `fps` filter rather than OpenCV's
frame-by-frame `VideoCapture.read()` loop in a tight Python loop — FFmpeg
decodes and writes frames in one subprocess call, which is both faster and
keeps this stage's failure mode identical to transcoder.py's (subprocess
exit code + stderr), rather than introducing a second failure shape.
"""
from __future__ import annotations

import asyncio
import glob
import os
import uuid

from services.preprocessing.core.config import Settings
from services.preprocessing.core.logging import get_logger
from services.preprocessing.models.schemas import FrameCandidate, PreprocessingError, PreprocessingStage

logger = get_logger(__name__)


class FrameExtractor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def extract(self, video_id: uuid.UUID, normalized_path: str) -> list[FrameCandidate]:
        frame_dir = os.path.join(self._settings.TMP_DIR, f"{video_id}_frames")
        os.makedirs(frame_dir, exist_ok=True)

        fps_filter = 1.0 / self._settings.FRAME_EXTRACTION_INTERVAL_SECONDS
        output_pattern = os.path.join(frame_dir, "%06d.jpg")

        cmd = [
            "ffmpeg", "-y",
            "-i", normalized_path,
            "-vf", f"fps={fps_filter}",
            "-qscale:v", "2",
            output_pattern,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._settings.FFMPEG_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            raise PreprocessingError(
                "FFmpeg frame extraction timed out",
                stage=PreprocessingStage.FRAME_EXTRACTION,
                recoverable=True,
            ) from exc

        if proc.returncode != 0:
            raise PreprocessingError(
                f"FFmpeg frame extraction failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace')[-2000:]}",
                stage=PreprocessingStage.FRAME_EXTRACTION,
                recoverable=False,
            )

        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        if not frame_paths:
            raise PreprocessingError(
                "FFmpeg produced zero frames — video may be empty or unreadable",
                stage=PreprocessingStage.FRAME_EXTRACTION,
                recoverable=False,
            )

        interval_ms = int(self._settings.FRAME_EXTRACTION_INTERVAL_SECONDS * 1000)
        return [
            FrameCandidate(
                sequence_index=i,
                timestamp_ms=i * interval_ms,
                local_path=path,
            )
            for i, path in enumerate(frame_paths)
        ]
