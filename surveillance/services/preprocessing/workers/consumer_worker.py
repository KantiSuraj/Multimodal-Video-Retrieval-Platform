from __future__ import annotations

import asyncio

from services.preprocessing.core.config import Settings
from services.preprocessing.core.logging import get_logger
from services.preprocessing.services.preprocessing import PreprocessingService
from services.preprocessing.services.queue import PreprocessingConsumer

logger = get_logger(__name__)


async def run_consumer_worker(
    settings: Settings,
    preprocessing_service: PreprocessingService,
) -> tuple[PreprocessingConsumer, asyncio.Task]:
    consumer = PreprocessingConsumer(
        settings, on_video_ingested=preprocessing_service.process_video
    )
    task = asyncio.create_task(consumer.start_consuming(), name="preprocessing_consumer")
    logger.info(
        "preprocessing_consumer_started",
        queue=settings.QUEUE_NAME,
        routing_key=settings.CONSUME_ROUTING_KEY,
    )
    return consumer, task

