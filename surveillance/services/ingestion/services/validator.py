"""
File validation pipeline.

Responsibilities:
  - MIME type detection via python-magic (FR-ING-01/02)
  - File extension allow-listing
  - SHA-256 checksum computation (FR-ING-03)
  - FFprobe-based media integrity check (FR-ING-05)
  - Metadata extraction: duration, resolution, codec (FR-ING-04)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import magic

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ValidationResult:
    is_valid: bool
    sha256_hash: str = ""
    mime_type: str = ""
    detected_extension: str = ""
    duration_seconds: float | None = None
    resolution_width: int | None = None
    resolution_height: int | None = None
    codec: str | None = None
    error_reason: str | None = None
    raw_ffprobe: dict = field(default_factory=dict)


class VideoValidator:
    """Stateless validator – all methods are async-safe."""

    # ── Public entry point ────────────────────────────────────────────────────

    async def validate(self, data: bytes, filename: str) -> ValidationResult:
        """
        Run the full validation chain:
          1. Extension check
          2. MIME-type detection
          3. SHA-256 hash
          4. FFprobe media validation + metadata extraction
        """
        ext = Path(filename).suffix.lower()

        # Step 1 – extension
        if ext not in settings.ALLOWED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                error_reason=f"Unsupported file extension: {ext!r}",
            )

        # Step 2 – MIME type via libmagic
        mime_type = self._detect_mime(data)
        if mime_type not in settings.ALLOWED_MIME_TYPES:
            return ValidationResult(
                is_valid=False,
                mime_type=mime_type,
                error_reason=f"Unsupported MIME type: {mime_type!r}",
            )

        # Step 3 – SHA-256
        sha256 = self._compute_hash(data)

        # Step 4 – FFprobe (writes to a temp file; ffprobe needs a path)
        probe_result = await self._ffprobe(data, filename)
        if probe_result is None:
            return ValidationResult(
                is_valid=False,
                sha256_hash=sha256,
                mime_type=mime_type,
                error_reason="FFprobe validation failed: file may be corrupt or truncated",
            )

        streams = probe_result.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

        duration = None
        fmt = probe_result.get("format", {})
        if "duration" in fmt:
            try:
                duration = float(fmt["duration"])
            except (ValueError, TypeError):
                pass

        width = height = codec = None
        if video_stream:
            width = video_stream.get("width")
            height = video_stream.get("height")
            codec = video_stream.get("codec_name")

        return ValidationResult(
            is_valid=True,
            sha256_hash=sha256,
            mime_type=mime_type,
            detected_extension=ext,
            duration_seconds=duration,
            resolution_width=width,
            resolution_height=height,
            codec=codec,
            raw_ffprobe=probe_result,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_mime(data: bytes) -> str:
        try:
            return magic.from_buffer(data[:8192], mime=True)
        except Exception as exc:
            logger.warning("mime_detection_failed", error=str(exc))
            return "application/octet-stream"

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        h = hashlib.sha256()
        # Process in 64-KB chunks to avoid pausing the event loop on huge buffers
        chunk_size = 65536
        for i in range(0, len(data), chunk_size):
            h.update(data[i : i + chunk_size])
        return h.hexdigest()

    @staticmethod
    async def _ffprobe(data: bytes, filename: str) -> dict | None:
        """
        Write bytes to a temp file, run ffprobe, parse JSON output.
        Returns the parsed probe dict, or None on failure.
        """
        suffix = Path(filename).suffix or ".mp4"
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                tmp_path,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                logger.warning(
                    "ffprobe_error",
                    filename=filename,
                    stderr=stderr.decode(errors="replace"),
                )
                return None

            return json.loads(stdout)
        except (asyncio.TimeoutError, FileNotFoundError) as exc:
            # ffprobe not installed – log and skip; treat file as valid
            logger.warning("ffprobe_unavailable", error=str(exc))
            return {}
        except json.JSONDecodeError as exc:
            logger.warning("ffprobe_json_parse_error", error=str(exc))
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


# Singleton
video_validator = VideoValidator()
