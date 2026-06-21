"""
services/preprocessing/services/transcoder.py — Stage 1: normalization

Mirrors the FFprobe subprocess pattern in ingestion's validator.py:
asyncio.create_subprocess_exec, wait with a timeout, parse exit code and
stderr on failure. FFmpeg is launched as a subprocess for the same reason
FFprobe is — it's the industry-standard tool and there's no reliable
pure-Python equivalent.

Output contract (fixed by the spec, not configurable per-call): H.264,
720p, 25fps. Encoder preset/CRF are deliberately not exposed as knobs the
caller can vary per-video — keeping the output format uniform is what lets
every downstream stage assume a single fps/resolution.
"""
from __future__ import annotations

import asyncio
import os
import uuid

from services.preprocessing.core.config import Settings
from services.preprocessing.core.logging import get_logger
from services.preprocessing.models.schemas import NormalizationResult, PreprocessingError, PreprocessingStage

logger = get_logger(__name__)


class VideoTranscoder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def normalize(self, video_id: uuid.UUID, source_path: str) -> NormalizationResult:
        width, height = self._settings.NORMALIZED_RESOLUTION.split("x")
        output_path = os.path.join(self._settings.TMP_DIR, f"{video_id}_normalized.mp4")
        os.makedirs(self._settings.TMP_DIR, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", source_path,
            "-vf", f"scale={width}:{height}",
            "-r", str(self._settings.NORMALIZED_FPS),
            "-c:v", self._settings.NORMALIZED_CODEC,
            "-pix_fmt", "yuv420p",
            "-an",
            output_path,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._settings.FFMPEG_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            raise PreprocessingError(
                f"FFmpeg normalization timed out after {self._settings.FFMPEG_TIMEOUT_SECONDS}s",
                stage=PreprocessingStage.NORMALIZATION,
                recoverable=True,
            ) from exc

        if proc.returncode != 0 or not os.path.exists(output_path):
            raise PreprocessingError(
                f"FFmpeg normalization failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace')[-2000:]}",
                stage=PreprocessingStage.NORMALIZATION,
                recoverable=False,  # bad/corrupt source video — not a transient infra issue
            )

        return NormalizationResult(
            local_path=output_path,
            width=int(width),
            height=int(height),
            fps=self._settings.NORMALIZED_FPS,
            duration_seconds=await self._probe_duration(output_path),
            codec=self._settings.NORMALIZED_CODEC,
        )

    async def _probe_duration(self, path: str) -> float:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return 0.0
        import json

        try:
            payload = json.loads(stdout)
            return float(payload["format"]["duration"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return 0.0
