"""
FastAPI application factory and lifespan.

Startup sequence:
  1. Configure structured logging
  2. Initialise storage buckets (MinIO)
  3. Start AMQP publisher
  4. Launch filesystem watcher as background task

Shutdown sequence:
  1. Cancel filesystem watcher
  2. Close AMQP connection
  3. Dispose SQLAlchemy engine
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from core.config import get_settings
from core.database import engine
from core.logging import configure_logging, get_logger
from services.queue import mq_publisher
from services.storage import storage_service
from workers.fs_watcher import start_filesystem_watcher

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    configure_logging()
    logger.info("startup_begin", version=settings.APP_VERSION)

    await storage_service.startup()
    await mq_publisher.startup()

    watcher_task = asyncio.create_task(
        start_filesystem_watcher(),
        name="fs_watcher",
    )
    logger.info("startup_complete")

    yield  # ── Application running ────────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
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
            "Video ingestion service: accepts uploads via HTTP multipart, RTSP pull, "
            "and local filesystem watch; validates, deduplicates, stores in MinIO, "
            "persists to PostgreSQL, and publishes domain events to RabbitMQ."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.DEBUG else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again."},
        )

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["Ops"])
    async def health():
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_config=None,  # structlog handles logging
    )