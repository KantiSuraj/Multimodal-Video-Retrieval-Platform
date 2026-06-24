"""Application entry point for the Search Service.

Responsibilities (per the bootstrap discipline):
    - startup
    - shutdown
    - dependency construction
    - configuration loading

No retrieval logic belongs here.

Pattern mirrors indexing/main.py: asynccontextmanager lifespan that
constructs dependencies, wires them together, and tears them down on
shutdown.  The search service has no consumer worker — it is request-
driven through HTTP APIs rather than event-driven through RabbitMQ.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.search.api.routes import router, set_search_service
from services.search.core.config import get_settings
from services.search.core.logging import configure_logging, get_logger
from services.search.services.cache import SearchCacheClient
from services.search.services.clip_encoder import CLIPQueryEncoder
from services.search.services.qdrant import QdrantSearchClient
from services.search.services.search import SearchService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(debug=settings.DEBUG)

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant = QdrantSearchClient(settings)
    await qdrant.startup()

    # ── Redis ─────────────────────────────────────────────────────────────────
    cache = SearchCacheClient(settings)
    await cache.startup()

    # ── CLIP encoder ──────────────────────────────────────────────────────────
    # Loading the CLIP model is CPU/GPU-bound and happens synchronously
    # at startup — same pattern as the Embedding Service.
    encoder = CLIPQueryEncoder(settings)
    encoder.load()

    # ── Service assembly ──────────────────────────────────────────────────────
    search_service = SearchService(settings, encoder, qdrant, cache)
    set_search_service(search_service)

    logger.info("search_service_started")
    try:
        yield
    finally:
        await qdrant.shutdown()
        await cache.shutdown()
        logger.info("search_service_stopped")


app = FastAPI(title="search-service", lifespan=lifespan)
app.include_router(router)
