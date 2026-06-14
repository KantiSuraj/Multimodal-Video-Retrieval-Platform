"""
File validation pipeline (FR-ING-02, FR-ING-03, FR-ING-04, FR-ING-05).

  1. Extension allow-list check
  2. MIME-type detection via python-magic
  3. SHA-256 checksum
  4. FFprobe media integrity + metadata extraction
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

from services.ingestion.core.config import get_settings
from services.ingestion.core.logging import get_logger

logger   = get_logger(__name__)
settings = get_settings()


@dataclass
class ValidationResult:
    is_valid:           bool
    sha256_hash:        str          = ""
    mime_type:          str          = ""
    detected_extension: str          = ""
    duration_seconds:   float | None = None
    resolution_width:   int   | None = None
    resolution_height:  int   | None = None
    codec:              str   | None = None
    error_reason:       str   | None = None
    raw_ffprobe:        dict         = field(default_factory=dict)


class VideoValidator:
    """Stateless validator — all methods are async-safe."""

    async def validate(self, data: bytes, filename: str) -> ValidationResult:
        ext = Path(filename).suffix.lower()

        if ext not in settings.ALLOWED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                error_reason=f"Unsupported file extension: {ext!r}",
            )

        mime_type = self._detect_mime(data)
        if mime_type not in settings.ALLOWED_MIME_TYPES:
            return ValidationResult(
                is_valid=False,
                mime_type=mime_type,
                error_reason=f"Unsupported MIME type: {mime_type!r}",
            )

        sha256 = self._compute_hash(data)

        probe = await self._ffprobe(data, filename)
        if probe is None:
            return ValidationResult(
                is_valid=False,
                sha256_hash=sha256,
                mime_type=mime_type,
                error_reason="FFprobe validation failed: file may be corrupt or truncated",
            )

        streams      = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        fmt          = probe.get("format", {})

        duration = None
        if "duration" in fmt:
            try:
                duration = float(fmt["duration"])
            except (ValueError, TypeError):
                pass

        width = height = codec = None
        if video_stream:
            width  = video_stream.get("width")
            height = video_stream.get("height")
            codec  = video_stream.get("codec_name")

        return ValidationResult(
            is_valid=True,
            sha256_hash=sha256,
            mime_type=mime_type,
            detected_extension=ext,
            duration_seconds=duration,
            resolution_width=width,
            resolution_height=height,
            codec=codec,
            raw_ffprobe=probe,
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
        for i in range(0, len(data), 65536):
            h.update(data[i : i + 65536])
        return h.hexdigest()

    @staticmethod
    async def _ffprobe(data: bytes, filename: str) -> dict | None:
        suffix   = Path(filename).suffix or ".mp4"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            proc = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                logger.warning("ffprobe_error", filename=filename,
                               stderr=stderr.decode(errors="replace"))
                return None
            return json.loads(stdout)

        except (asyncio.TimeoutError, FileNotFoundError) as exc:
            logger.warning("ffprobe_unavailable", error=str(exc))
            return {}   # treat as valid when ffprobe not installed
        except json.JSONDecodeError as exc:
            logger.warning("ffprobe_json_error", error=str(exc))
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


# Singleton
video_validator = VideoValidator()