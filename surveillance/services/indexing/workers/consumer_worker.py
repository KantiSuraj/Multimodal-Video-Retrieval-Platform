"""Runtime harness only. Wires the queue layer to the orchestrator.

Returns both the consumer (so main.py can call stop() on shutdown) and the
task running start_consuming() (so main.py can cancel it).
"""
from __future__ import annotations

import asyncio

from services.indexing.core.config import get_settings
from services.indexing.core.logging import get_logger
from services.indexing.services.indexing import IndexingService
from services.indexing.services.queue import IndexingConsumer

logger = get_logger(__name__)


async def run_consumer_worker(
    indexing_service: IndexingService,
) -> tuple[IndexingConsumer, asyncio.Task]:
    settings = get_settings()
    consumer = IndexingConsumer(settings, on_embeddings_ready=indexing_service.process_embeddings)
    task = asyncio.create_task(consumer.start_consuming())
    logger.info("indexing_consumer_worker_started")
    return consumer, task
