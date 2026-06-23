"""Runtime harness only. Wires the queue layer to the orchestrator.

Returns both the consumer (so main.py can call stop() on shutdown) and the
task running start_consuming() (so main.py can cancel it).
"""
from __future__ import annotations

import asyncio

from services.embedding.core.config import get_settings
from services.embedding.core.logging import get_logger
from services.embedding.services.embedding import EmbeddingService
from services.embedding.services.queue import EmbeddingConsumer

logger = get_logger(__name__)


async def run_consumer_worker(
    embedding_service: EmbeddingService,
) -> tuple[EmbeddingConsumer, asyncio.Task]:
    settings = get_settings()
    consumer = EmbeddingConsumer(settings, on_detection_complete=embedding_service.process_detections)
    task = asyncio.create_task(consumer.start_consuming())
    logger.info("embedding_consumer_worker_started")
    return consumer, task
