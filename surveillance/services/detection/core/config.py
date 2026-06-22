from functools import lru_cache

from shared.shared.config.base import BaseServiceSettings


class Settings(BaseServiceSettings):
    GROUNDING_DINO_MODEL_NAME: str = "IDEA-Research/grounding-dino-tiny"
    GROUNDING_DINO_DEVICE: str = "cuda"
    GROUNDING_DINO_TEXT_PROMPT: str = "person. car. backpack. bag. vehicle. face."
    GROUNDING_DINO_BOX_THRESHOLD: float = 0.35
    GROUNDING_DINO_TEXT_THRESHOLD: float = 0.25
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.4
    DETECTION_INFERENCE_TIMEOUT_SECONDS: int = 60

    MINIO_PROCESSED_FRAMES_BUCKET: str = "processed-frames"
    MINIO_DETECTION_CROPS_BUCKET: str = "detection-crops"
    MINIO_QUARANTINE_DETECTION_BUCKET: str = "quarantine-detection"

    DETECTION_QUEUE_NAME: str = "detection.tasks"
    DETECTION_CONSUME_ROUTING_KEY: str = "video.frames_extracted"
    DETECTION_PUBLISH_ROUTING_KEY: str = "video.detection_complete"
    # GPU-bound consumer — default BaseConsumer prefetch (10) is tuned for
    # lightweight I/O-bound work, not model inference. Keep this low so a
    # single replica never has more unacked messages than it can actually
    # run inference on concurrently.
    DETECTION_CONSUMER_PREFETCH: int = 2

    DETECTION_TMP_DIR: str = "/tmp/detection"


@lru_cache
def get_settings() -> Settings:
    return Settings()