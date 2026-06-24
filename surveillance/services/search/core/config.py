from functools import lru_cache

from shared.shared.config.base import BaseServiceSettings


class Settings(BaseServiceSettings):
    # ── CLIP model ────────────────────────────────────────────────────────────
    # Must match the model used during indexing so query vectors are
    # compatible with stored vectors.  A mismatch here is a fatal
    # configuration error — dimension validation catches it at startup.
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    CLIP_DEVICE: str = "cpu"
    SEARCH_INFERENCE_TIMEOUT_SECONDS: int = 30

    # ── Qdrant ────────────────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str | None = None
    QDRANT_USE_GRPC: bool = True
    QDRANT_TIMEOUT: float = 30.0
    QDRANT_COLLECTION_NAME: str = "surveillance_embeddings"
    QDRANT_VECTOR_DIMENSION: int = 512

    # ── Search parameters ─────────────────────────────────────────────────────
    # Search-time HNSW ef — must not be hardcoded anywhere else.
    SEARCH_HNSW_EF: int = 128
    SEARCH_TOP_K: int = 50  # ANN candidates before deduplication

    # ── Temporal deduplication ────────────────────────────────────────────────
    SEARCH_TEMPORAL_DEDUP_WINDOW_MS: int = 5000  # 5 seconds

    # ── Redis caching ─────────────────────────────────────────────────────────
    # Query embedding cache: avoid re-encoding identical text/image queries.
    SEARCH_EMBEDDING_CACHE_TTL_SECONDS: int = 3600  # 1 hour
    # Search result cache: avoid re-running retrieval for identical queries+filters.
    SEARCH_RESULT_CACHE_TTL_SECONDS: int = 300  # 5 minutes

    # ── MinIO (thumbnail presigned URLs) ─────────────────────────────────────
    MINIO_PROCESSED_FRAMES_BUCKET: str = "processed-frames"
    MINIO_THUMBNAIL_PRESIGN_TTL_SECONDS: int = 3600

    # ── Pagination ────────────────────────────────────────────────────────────
    SEARCH_DEFAULT_PAGE_SIZE: int = 20
    SEARCH_MAX_PAGE_SIZE: int = 100

    # ── Crop detection (optional crop-based image search) ─────────────────────
    SEARCH_CROP_DETECTION_ENABLED: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
