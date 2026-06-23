"""RabbitMQ publisher + consumer for embedding.

Same contract as detection/services/queue.py: handle_message takes
(body, routing_key), matching the abstract method BaseConsumer actually
defines; start_consuming()/stop() are the real lifecycle methods on
BaseConsumer.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import ValidationError

from shared.shared.events.detection_complete import DetectionCompleteEvent
from shared.shared.events.embeddings_ready import EmbeddingsReadyEvent
from shared.shared.queue.consumer import BaseConsumer
from shared.shared.queue.publisher import BasePublisher

from services.embedding.core.config import Settings
from services.embedding.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingPublisher(BasePublisher):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._settings = settings

    async def publish_embeddings_ready(self, event: EmbeddingsReadyEvent) -> None:
        await self.publish(event, routing_key=self._settings.EMBEDDING_PUBLISH_ROUTING_KEY)


class EmbeddingConsumer(BaseConsumer):
    def __init__(
        self,
        settings: Settings,
        on_detection_complete: Callable[[DetectionCompleteEvent], Awaitable[None]],
    ):
        super().__init__(
            settings,
            queue_name=settings.EMBEDDING_QUEUE_NAME,
            routing_key=settings.EMBEDDING_CONSUME_ROUTING_KEY,
            prefetch=settings.EMBEDDING_CONSUMER_PREFETCH,
        )
        self._on_detection_complete = on_detection_complete

    async def handle_message(self, body: bytes, routing_key: str) -> None:
        try:
            event = DetectionCompleteEvent.model_validate_json(body)
        except ValidationError as exc:
            # A malformed event will never become valid on redelivery — this
            # is the message-level equivalent of a non-recoverable failure.
            # Log and return normally so message.process() acks rather than
            # nacking it into an infinite redelivery loop.
            logger.error(
                "embedding_malformed_event_dropped",
                routing_key=routing_key,
                reason=str(exc),
            )
            return

        logger.info(
            "embedding_message_received",
            video_id=str(event.video_id),
            frame_count=len(event.frames),
        )
        await self._on_detection_complete(event)
