"""RabbitMQ publisher + consumer for detection.

Fixed against the real shared.queue.consumer.BaseConsumer contract:
- handle_message takes (body, routing_key), matching the abstract method
  BaseConsumer actually defines and actually calls.
- start_consuming()/stop() are the real lifecycle methods — there is no
  startup()/consume() pair on BaseConsumer.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import ValidationError

from shared.shared.events.detection_complete import DetectionCompleteEvent
from shared.shared.events.frame_extracted import FramesExtractedEvent
from shared.shared.queue.consumer import BaseConsumer
from shared.shared.queue.publisher import BasePublisher

from services.detection.core.config import Settings
from services.detection.core.logging import get_logger

logger = get_logger(__name__)


class DetectionPublisher(BasePublisher):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._settings = settings

    async def publish_detection_complete(self, event: DetectionCompleteEvent) -> None:
        await self.publish(event, routing_key=self._settings.DETECTION_PUBLISH_ROUTING_KEY)


class DetectionConsumer(BaseConsumer):
    def __init__(
        self,
        settings: Settings,
        on_frames_extracted: Callable[[FramesExtractedEvent], Awaitable[None]],
    ):
        super().__init__(
            settings,
            queue_name=settings.DETECTION_QUEUE_NAME,
            routing_key=settings.DETECTION_CONSUME_ROUTING_KEY,
            prefetch=settings.DETECTION_CONSUMER_PREFETCH,
        )
        self._on_frames_extracted = on_frames_extracted

    async def handle_message(self, body: bytes, routing_key: str) -> None:
        try:
            event = FramesExtractedEvent.model_validate_json(body)
        except ValidationError as exc:
            # A malformed event will never become valid on redelivery — this
            # is the message-level equivalent of a non-recoverable failure.
            # Log and return normally so message.process() acks rather than
            # nacking it into an infinite redelivery loop.
            logger.error(
                "detection_malformed_event_dropped",
                routing_key=routing_key,
                reason=str(exc),
            )
            return

        logger.info(
            "detection_message_received",
            video_id=str(event.video_id),
            frame_count=len(event.frames),
        )
        await self._on_frames_extracted(event)