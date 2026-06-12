"""
Filesystem watcher worker (FR-ING-01 – local filesystem watch).

Uses the `watchdog` library to detect new files dropped into a watched
directory and feeds them into the standard ingestion pipeline.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from core.config import get_settings
from core.database import AsyncSessionLocal
from core.logging import get_logger
from services.ingestion import IngestionError, ingestion_service

logger = get_logger(__name__)
settings = get_settings()


class VideoFileHandler(FileSystemEventHandler):
    """Watchdog handler – enqueues new video files for ingestion."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        super().__init__()
        self._loop = loop
        self._queue = queue

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in settings.ALLOWED_EXTENSIONS:
            logger.info("file_detected", path=str(path))
            asyncio.run_coroutine_threadsafe(
                self._queue.put(str(path)), self._loop
            )


async def process_queue(queue: asyncio.Queue) -> None:
    """Consume file paths from the queue and run the ingestion pipeline."""
    while True:
        file_path: str = await queue.get()
        logger.info("processing_watched_file", path=file_path)
        try:
            async with AsyncSessionLocal() as db:
                result = await ingestion_service.ingest_filesystem(
                    db=db,
                    file_path=file_path,
                    camera_id=None,
                    location=settings.WATCH_DIRECTORY,
                    recorded_at=None,
                )
                await db.commit()
            logger.info("watched_file_ingested", path=file_path, video_id=str(result.video_id))
        except IngestionError as exc:
            logger.error("watched_file_ingestion_error", path=file_path, error=str(exc))
        except Exception as exc:
            logger.exception("watched_file_unexpected_error", path=file_path, error=str(exc))
        finally:
            queue.task_done()


async def start_filesystem_watcher() -> None:
    """
    Start watching the configured directory.
    Runs indefinitely; call from the app lifespan background task.
    """
    watch_dir = Path(settings.WATCH_DIRECTORY)
    watch_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()

    handler = VideoFileHandler(loop, queue)
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()

    logger.info("filesystem_watcher_started", directory=str(watch_dir))

    try:
        await process_queue(queue)
    finally:
        observer.stop()
        observer.join()
        logger.info("filesystem_watcher_stopped")
