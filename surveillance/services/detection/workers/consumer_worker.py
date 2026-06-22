"""Runtime harness only. Wires the queue layer to the orchestrator.

Returns both the consumer (so main.py can call stop() on shutdown) and the
task running start_consuming() (so main.py can cancel it).
"""
from __future__ import annotations

import asyncio

from services.detection.core.config import get_settings
from services.detection.core.logging import get_logger
from services.detection.services.detection import DetectionService
from services.detection.services.queue import DetectionConsumer

logger = get_logger(__name__)


async def run_consumer_worker(
    detection_service: DetectionService,
) -> tuple[DetectionConsumer, asyncio.Task]:
    settings = get_settings()
    consumer = DetectionConsumer(settings, on_frames_extracted=detection_service.process_frames)
    task = asyncio.create_task(consumer.start_consuming())
    logger.info("detection_consumer_worker_started")
    return consumer, task