"""
services/preprocessing/main.py

Mirrors ingestion's main.py lifecycle shape: FastAPI app whose lifespan
wires up infrastructure clients, starts background processing, and tears
it down on shutdown. The HTTP surface itself is just /health — all real
work happens in the consumer task started here.

Startup order (matches ingestion's dependency order: config -> logging ->
storage -> queue -> background processing):
  1. load settings
  2. configure logging
  3. construct ObjectStorageClient, ensure buckets exist
  4. construct + start PreprocessingPublisher
  5. construct PreprocessingService (business logic, framework-agnostic)
  6. start the RabbitMQ consumer as a background task

Shutdown order is the reverse: stop consuming, close publisher connection.
"""
from __future__ import annotations
import asyncio

from contextlib import asynccontextmanager

from fastapi import FastAPI

from shared.shared.storage.client import ObjectStorageClient

from services.preprocessing.api.routes import router
from services.preprocessing.core.config import get_settings
from services.preprocessing.core.logging import configure_logging, get_logger
from services.preprocessing.services.preprocessing import PreprocessingService
from services.preprocessing.services.queue import PreprocessingPublisher
from services.preprocessing.services.storage import PreprocessingStorageService
from services.preprocessing.workers.consumer_worker import run_consumer_worker

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(debug=settings.DEBUG)

    storage_client = ObjectStorageClient(settings)
    storage_service = PreprocessingStorageService(settings, storage_client)
    await storage_service.ensure_buckets()

    publisher = PreprocessingPublisher(settings)
    await publisher.startup()

    preprocessing_service = PreprocessingService(settings, storage_service, publisher)
    print("MAIN FILE VERSION 2026-06-21")
    consumer, consumer_task = await run_consumer_worker(settings, preprocessing_service)

    logger.info("preprocessing_service_started")
    try:
        yield
    finally:
        await consumer.stop()
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
        await publisher.shutdown()
        logger.info("preprocessing_service_stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="Preprocessing Service", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
