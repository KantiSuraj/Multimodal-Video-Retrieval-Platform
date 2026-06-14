"""
FastAPI application factory and lifespan for the ingestion service.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.ingestion.api.routes   import router
from services.ingestion.core.config  import get_settings
from services.ingestion.core.logging import configure_logging, get_logger
from services.ingestion.db.database  import engine
from services.ingestion.services.queue   import mq_publisher
from services.ingestion.services.storage import storage_service
from services.ingestion.workers.fs_watcher import start_filesystem_watcher

settings = get_settings()
logger   = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(debug=settings.DEBUG)
    logger.info("startup_begin", service=settings.APP_NAME, version=settings.APP_VERSION)

    await storage_service.startup()
    await mq_publisher.startup()

    watcher_task = asyncio.create_task(start_filesystem_watcher(), name="fs_watcher")
    logger.info("startup_complete")

    yield

    logger.info("shutdown_begin")
    watcher_task.cancel()
    try:
        await watcher_task
    except asyncio.CancelledError:
        pass

    await mq_publisher.shutdown()
    await engine.dispose()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Accepts video via HTTP multipart, RTSP pull, and local filesystem watch. "
            "Validates, deduplicates, stores in MinIO, persists to PostgreSQL, "
            "and publishes VideoIngestedEvent to RabbitMQ."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.DEBUG else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(status_code=500,
                            content={"detail": "An internal error occurred."})

    @app.get("/health", tags=["Ops"])
    async def health():
        return {"status": "ok", "service": settings.APP_NAME, "version": settings.APP_VERSION}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.ingestion.main:app", host="0.0.0.0", port=8000,
                reload=settings.DEBUG, log_config=None)