"""
services/preprocessing/services/queue.py

Mirrors services/ingestion/services/queue.py. Two thin subclasses of the
shared base classes:

  PreprocessingPublisher — one typed method, publish_frames_extracted()
  PreprocessingConsumer  — binds `preprocessing.tasks` to `video.ingested`,
                           delegates each message to the orchestrator

Connection management, channel setup, exchange declaration, ack/nack are
all inherited from shared.queue — nothing reimplemented here.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from shared.shared.events.frame_extracted import FramesExtractedEvent
from shared.shared.events.video_ingested import VideoIngestedEvent
from shared.shared.queue.consumer import BaseConsumer
from shared.shared.queue.publisher import BasePublisher

from services.preprocessing.core.config import Settings
from services.preprocessing.core.logging import get_logger

logger = get_logger(__name__)


class PreprocessingPublisher(BasePublisher):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._settings = settings

    async def publish_frames_extracted(self, event: FramesExtractedEvent) -> None:
        await self.publish(event, routing_key=self._settings.PUBLISH_ROUTING_KEY)

class PreprocessingConsumer(BaseConsumer):
    """Binds to `video.ingested`. Each message is parsed into a
    VideoIngestedEvent and handed to the injected handler."""

    def __init__(
        self,
        settings: Settings,
        on_video_ingested: Callable[[VideoIngestedEvent], Awaitable[None]],
    ) -> None:
        super().__init__(
            settings,
            queue_name=settings.QUEUE_NAME,
            routing_key=settings.CONSUME_ROUTING_KEY,
            prefetch=1,  # CPU/FFmpeg-bound, single in-flight message — see Medium #8
        )
        self._on_video_ingested = on_video_ingested

    async def handle_message(self, body: bytes, routing_key: str) -> None:
        event = VideoIngestedEvent.model_validate_json(body)
        logger.info("preprocessing_message_received", video_id=str(event.video_id))
        logger.info(
            "starting_preprocessing",
             video_id=str(event.video_id),
        )
        await self._on_video_ingested(event)
        logger.info(
            "preprocessing_completed",
            video_id=str(event.video_id),
        )