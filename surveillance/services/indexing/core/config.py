from functools import lru_cache

from shared.shared.config.base import BaseServiceSettings


class Settings(BaseServiceSettings):
    # ── Qdrant ────────────────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str | None = None
    QDRANT_USE_GRPC: bool = True
    QDRANT_TIMEOUT: float = 30.0

    # ── Collection schema ─────────────────────────────────────────────────────
    QDRANT_COLLECTION_NAME: str = "surveillance_embeddings"
    QDRANT_VECTOR_DIMENSION: int = 768
    QDRANT_DISTANCE_METRIC: str = "Cosine"  # "Cosine" | "Euclid" | "Dot"

    # ── HNSW ──────────────────────────────────────────────────────────────────
    QDRANT_HNSW_M: int = 16
    QDRANT_HNSW_EF_CONSTRUCT: int = 100

    # ── Batching ──────────────────────────────────────────────────────────────
    QDRANT_UPSERT_BATCH_SIZE: int = 100

    # ── Consumer ──────────────────────────────────────────────────────────────
    INDEXING_QUEUE_NAME: str = "indexing.tasks"
    INDEXING_CONSUME_ROUTING_KEY: str = "video.embeddings_ready"
    # Indexing is I/O-bound (Qdrant + DB writes), not GPU-bound, so a
    # moderate prefetch is appropriate.
    INDEXING_CONSUMER_PREFETCH: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
