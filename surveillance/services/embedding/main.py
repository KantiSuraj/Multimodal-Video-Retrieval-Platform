from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from shared.shared.storage.client import ObjectStorageClient

from services.embedding.api.routes import router
from services.embedding.core.config import get_settings
from services.embedding.core.logging import configure_logging, get_logger
from services.embedding.services.clip_model import CLIPEmbedder
from services.embedding.services.embedding import EmbeddingService
from services.embedding.services.queue import EmbeddingPublisher
from services.embedding.services.storage import EmbeddingStorageService
from services.embedding.workers.consumer_worker import run_consumer_worker

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(debug=settings.DEBUG)

    storage_client = ObjectStorageClient(settings)
    # Embedding only reads from processed-frames / detection-crops — both
    # are ensured by preprocessing/detection respectively, so there is
    # nothing for embedding to ensure_bucket() on here.

    embedder = CLIPEmbedder(settings)
    embedder.load()

    storage = EmbeddingStorageService(storage_client, settings)
    publisher = EmbeddingPublisher(settings)
    await publisher.startup()

    embedding_service = EmbeddingService(settings, embedder, storage, publisher)
    consumer, consumer_task = await run_consumer_worker(embedding_service)

    logger.info("embedding_service_started")
    try:
        yield
    finally:
        consumer_task.cancel()
        await consumer.stop()
        await publisher.shutdown()
        logger.info("embedding_service_stopped")


app = FastAPI(title="embedding-service", lifespan=lifespan)
app.include_router(router)
