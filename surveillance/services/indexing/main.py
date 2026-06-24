from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.indexing.api.routes import router
from services.indexing.core.config import get_settings
from services.indexing.core.logging import configure_logging, get_logger
from services.indexing.services.indexing import IndexingService
from services.indexing.services.qdrant import QdrantService
from services.indexing.workers.consumer_worker import run_consumer_worker

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(debug=settings.DEBUG)

    qdrant = QdrantService(settings)
    await qdrant.startup()
    await qdrant.ensure_collection()

    indexing_service = IndexingService(settings, qdrant)
    consumer, consumer_task = await run_consumer_worker(indexing_service)

    logger.info("indexing_service_started")
    try:
        yield
    finally:
        consumer_task.cancel()
        await consumer.stop()
        await qdrant.shutdown()
        logger.info("indexing_service_stopped")


app = FastAPI(title="indexing-service", lifespan=lifespan)
app.include_router(router)
