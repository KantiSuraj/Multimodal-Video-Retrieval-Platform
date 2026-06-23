from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum


class EmbeddingStage(str, Enum):
    FETCH_ARTIFACT = "FETCH_ARTIFACT"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    PERSIST = "PERSIST"
    PUBLISH = "PUBLISH"


class EmbeddingError(Exception):
    def __init__(self, message: str, stage: EmbeddingStage, recoverable: bool):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.recoverable = recoverable


@dataclass
class ArtifactRef:
    """One visual artifact (a frame or a detection crop) to be embedded.

    kind is "frame" | "crop" — embedding owns no other artifact types.
    detection_id is only present for crop artifacts; carried through so a
    future caller could correlate a crop embedding back to its detection
    without a second lookup, but is not currently persisted as its own
    column (EmbeddingRecord has no detection_id field).
    """

    kind: str
    source_path: str
    source_bucket: str
    timestamp_ms: int | None
    label: str | None
    detection_id: uuid.UUID | None = None


@dataclass
class PersistedEmbedding:
    """One artifact paired with its generated, L2-normalised vector."""

    artifact: ArtifactRef
    vector: list[float]
