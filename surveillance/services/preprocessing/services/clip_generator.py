"""
services/preprocessing/services/clip_generator.py — Stage 6: clip generation

For each scene boundary, cuts a clip from the normalized video using
FFmpeg's stream copy (`-c copy`) where possible for speed, falling back
implicitly to re-encode only if a scene is shorter than the minimum clip
duration and needs padding. Clip duration is clamped to
[MIN_CLIP_DURATION_SECONDS, MAX_CLIP_DURATION_SECONDS] — scenes longer
than the max are split into consecutive clips; scenes shorter than the
default are extended symmetrically (without running past the video's
own duration) rather than upscaled artificially.

Uses `-ss`/`-t` for fast seeking on the already-normalized (and therefore
keyframe-friendly, since FFmpeg re-encoded it with default GOP settings)
video — this is materially cheaper than re-decoding the entire video per
clip.
"""
from __future__ import annotations

import asyncio
import os
import uuid

from services.preprocessing.core.config import Settings
from services.preprocessing.core.logging import get_logger
from services.preprocessing.models.schemas import ClipSpec, PreprocessingError, PreprocessingStage, SceneSegment

logger = get_logger(__name__)


class ClipGenerator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _clip_windows(self, scene: SceneSegment, video_duration_ms: int) -> list[tuple[int, int]]:
        min_ms = int(self._settings.MIN_CLIP_DURATION_SECONDS * 1000)
        max_ms = int(self._settings.MAX_CLIP_DURATION_SECONDS * 1000)
        target_ms = int(self._settings.DEFAULT_CLIP_DURATION_SECONDS * 1000)

        duration = scene.end_ms - scene.start_ms

        if duration < min_ms:
            # extend symmetrically up to target length, clamped to video bounds
            wanted = max(min_ms, min(target_ms, duration))
            pad = (wanted - duration) // 2
            start = max(0, scene.start_ms - pad)
            end = min(video_duration_ms, start + wanted)
            return [(start, end)]

        if duration <= max_ms:
            return [(scene.start_ms, scene.end_ms)]

        # split long scenes into consecutive max_ms windows
        windows = []
        cursor = scene.start_ms
        while cursor < scene.end_ms:
            end = min(cursor + max_ms, scene.end_ms)
            windows.append((cursor, end))
            cursor = end
        return windows

    async def generate(
        self,
        video_id: uuid.UUID,
        normalized_path: str,
        scenes: list[SceneSegment],
        video_duration_ms: int,
    ) -> list[ClipSpec]:
        clip_dir = os.path.join(self._settings.TMP_DIR, f"{video_id}_clips")
        os.makedirs(clip_dir, exist_ok=True)

        clips: list[ClipSpec] = []
        for scene in scenes:
            for window_index, (start_ms, end_ms) in enumerate(
                self._clip_windows(scene, video_duration_ms)
            ):
                out_path = os.path.join(clip_dir, f"scene_{scene.scene_id:04d}_{window_index}.mp4")
                clip = await self._cut(normalized_path, start_ms, end_ms, out_path, scene.scene_id)
                clips.append(clip)
        return clips

    async def _cut(
        self, source_path: str, start_ms: int, end_ms: int, out_path: str, scene_id: int
    ) -> ClipSpec:
        duration_seconds = (end_ms - start_ms) / 1000.0
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_ms / 1000.0:.3f}",
            "-i", source_path,
            "-t", f"{duration_seconds:.3f}",
            "-c", "copy",
            out_path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0 or not os.path.exists(out_path):
            raise PreprocessingError(
                f"FFmpeg clip generation failed for scene {scene_id} "
                f"(exit {proc.returncode}): {stderr.decode(errors='replace')[-1000:]}",
                stage=PreprocessingStage.CLIP_GENERATION,
                recoverable=False,
            )

        return ClipSpec(
            scene_id=scene_id,
            start_ms=start_ms,
            end_ms=end_ms,
            local_path=out_path,
            duration_seconds=duration_seconds,
        )
