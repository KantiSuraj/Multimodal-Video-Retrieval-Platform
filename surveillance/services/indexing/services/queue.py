"""RabbitMQ consumer for indexing.

Same contract as embedding/services/queue.py: handle_message takes
(body, routing_key), matching the abstract method BaseConsumer actually
defines; start_consuming()/stop() are the real lifecycle methods on
BaseConsumer.

Indexing does not publish events — it is a terminal consumer in the
pipeline.  If a downstream publish step is ever needed (e.g.
IndexingCompleteEvent for a monitoring dashboard), a publisher can be
added here following the same BasePublisher pattern as embedding.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import ValidationError

from shared.shared.events.embeddings_ready import EmbeddingsReadyEvent
from shared.shared.queue.consumer import BaseConsumer

from services.indexing.core.config import Settings
from services.indexing.core.logging import get_logger

logger = get_logger(__name__)


class IndexingConsumer(BaseConsumer):
    def __init__(
        self,
        settings: Settings,
        on_embeddings_ready: Callable[[EmbeddingsReadyEvent], Awaitable[None]],
    ):
        super().__init__(
            settings,
            queue_name=settings.INDEXING_QUEUE_NAME,
            routing_key=settings.INDEXING_CONSUME_ROUTING_KEY,
            prefetch=settings.INDEXING_CONSUMER_PREFETCH,
        )
        self._on_embeddings_ready = on_embeddings_ready

    async def handle_message(self, body: bytes, routing_key: str) -> None:
        try:
            event = EmbeddingsReadyEvent.model_validate_json(body)
        except ValidationError as exc:
            # A malformed event will never become valid on redelivery — this
            # is the message-level equivalent of a non-recoverable failure.
            # Log and return normally so message.process() acks rather than
            # nacking it into an infinite redelivery loop.
            logger.error(
                "indexing_malformed_event_dropped",
                routing_key=routing_key,
                reason=str(exc),
            )
            return

        logger.info(
            "indexing_message_received",
            video_id=event.video_id,
            embedding_count=len(event.embeddings),
            model_name=event.model_name,
        )
        await self._on_embeddings_ready(event)
