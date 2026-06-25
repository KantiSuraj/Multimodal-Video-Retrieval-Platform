from functools import lru_cache

from shared.shared.config.base import BaseServiceSettings


class Settings(BaseServiceSettings):
    CLIP_MODEL_NAME: str = "openai/clip-vit-large-patch14"
    CLIP_DEVICE: str = "cuda"
    EMBEDDING_INFERENCE_TIMEOUT_SECONDS: int = 60

    # Same buckets detection already reads from — preprocessing owns
    # processed-frames, detection owns detection-crops. Embedding only
    # reads from both; it never writes to either.
    MINIO_PROCESSED_FRAMES_BUCKET: str = "processed-frames"
    MINIO_DETECTION_CROPS_BUCKET: str = "detection-crops"

    EMBEDDING_QUEUE_NAME: str = "embedding.tasks"
    EMBEDDING_CONSUME_ROUTING_KEY: str = "video.detection_complete"
    EMBEDDING_PUBLISH_ROUTING_KEY: str = "video.embeddings_ready"
    # GPU-bound consumer — same rationale as DETECTION_CONSUMER_PREFETCH: the
    # default BaseConsumer prefetch (10) is tuned for lightweight I/O-bound
    # work, not model inference. Keep this low so a single replica never has
    # more unacked messages than it can actually run inference on
    # concurrently.
    EMBEDDING_CONSUMER_PREFETCH: int = 2

    # Max embeddings per RabbitMQ message.  Each embedding carries a 512-float
    # vector (~3 KB JSON); 50 per message ≈ 150 KB, well under the aio_pika
    # default frame limit.  Reduce if you switch to a larger CLIP model
    # (e.g. clip-vit-large-patch14 → 768-dim → ~4.5 KB/embedding).
    EMBEDDING_PUBLISH_BATCH_SIZE: int = 50

    EMBEDDING_TMP_DIR: str = "/tmp/embedding"


@lru_cache
def get_settings() -> Settings:
    return Settings()
