from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from shared.shared.storage.client import ObjectStorageClient

from services.detection.api.routes import router
from services.detection.core.config import get_settings
from services.detection.core.logging import configure_logging, get_logger
from services.detection.services.detection import DetectionService
from services.detection.services.grounding_dino import GroundingDINODetector
from services.detection.services.queue import DetectionPublisher
from services.detection.services.storage import DetectionStorageService
from services.detection.workers.consumer_worker import run_consumer_worker

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(debug=settings.DEBUG)

    storage_client = ObjectStorageClient(settings)
    await storage_client.ensure_bucket(settings.MINIO_DETECTION_CROPS_BUCKET)
    await storage_client.ensure_bucket(settings.MINIO_QUARANTINE_DETECTION_BUCKET)

    detector = GroundingDINODetector(settings)
    detector.load()

    storage = DetectionStorageService(storage_client, settings)
    publisher = DetectionPublisher(settings)
    await publisher.startup()

    detection_service = DetectionService(settings, detector, storage, publisher)
    consumer, consumer_task = await run_consumer_worker(detection_service)

    logger.info("detection_service_started")
    try:
        yield
    finally:
        consumer_task.cancel()
        await consumer.stop()
        await publisher.shutdown()
        logger.info("detection_service_stopped")


app = FastAPI(title="detection-service", lifespan=lifespan)
app.include_router(router)