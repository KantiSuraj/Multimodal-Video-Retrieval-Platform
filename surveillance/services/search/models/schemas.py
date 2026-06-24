"""Internal schemas for the Search Service.

Mirrors the pattern from embedding/models/schemas.py and
indexing/models/schemas.py: typed dataclasses for internal data flow,
a stage enum for structured error attribution, and a domain error class
carrying both a human-readable message and a recoverable flag so
callers can decide between a logged-and-swallowed failure path and a
user-visible 4xx/5xx response.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SearchStage(str, Enum):
    VALIDATION = "VALIDATION"
    ENCODE = "ENCODE"
    FILTER_BUILD = "FILTER_BUILD"
    ANN_RETRIEVE = "ANN_RETRIEVE"
    HYDRATE = "HYDRATE"
    RANK = "RANK"
    DEDUPLICATE = "DEDUPLICATE"
    PAGINATE = "PAGINATE"
    CACHE = "CACHE"


class SearchError(Exception):
    """Mirrors EmbeddingError / IndexingError.

    recoverable=True  → transient failure (Qdrant/Redis/PG down); the
                        HTTP handler returns 503 and the client retries.
    recoverable=False → permanent failure (bad request, bad token, bad
                        dimension); the HTTP handler returns 4xx and
                        there is no point retrying.
    """

    def __init__(self, message: str, stage: SearchStage, recoverable: bool) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.recoverable = recoverable


@dataclass
class MetadataFilters:
    """Parsed, validated metadata filter specification.

    All fields are optional — omitting a field means no filtering on
    that dimension.  Future label filtering fits naturally here.
    """

    camera_ids: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    start_ms: int | None = None   # epoch-ms of the earliest acceptable frame
    end_ms: int | None = None     # epoch-ms of the latest acceptable frame
    labels: list[str] = field(default_factory=list)  # future-compatible


@dataclass
class RetrievedPoint:
    """One raw ANN result from Qdrant before metadata hydration.

    Carries only the data Qdrant returns — the orchestrator enriches it
    into a HydratedResult after PostgreSQL hydration.
    """

    point_id: str
    score: float
    payload: dict[str, Any]


@dataclass
class HydratedResult:
    """One fully-enriched search result ready for the response serialiser.

    Fields mirror the response contract specified in the architecture
    document.  thumbnail_url is presigned from MinIO and may be None if
    the source path is unavailable.
    """

    video_id: str
    clip_id: str                        # EmbeddingRecord.id (str representation)
    camera_name: str | None
    camera_location: str | None
    timestamp_start: int                # ms from video start
    timestamp_end: int                  # ms from video start (start + clip_length)
    similarity_score: float
    thumbnail_url: str | None
    detected_labels: list[str]
    video_start_epoch: int | None       # recorded_at as epoch-ms


@dataclass
class SearchHistoryEntry:
    """One record in the search history log.

    Stored in Redis as JSON.  The query_type distinguishes text vs image
    searches for the GET /history filter support.
    """

    id: str
    query_type: str                     # "text" | "image" | "multi_image"
    query_text: str | None
    filters: dict[str, Any]
    result_count: int
    created_at: str                     # ISO-8601
