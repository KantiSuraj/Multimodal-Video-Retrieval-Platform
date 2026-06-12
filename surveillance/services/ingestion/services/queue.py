"""
Async AMQP publisher using aio-pika.

Publishes domain events to RabbitMQ.  The exchange is declared as
`topic` so downstream consumers can subscribe with routing-key patterns
such as `video.*` or `video.ingested`.
"""
from __future__ import annotations

import json

import aio_pika
from aio_pika import ExchangeType, Message

from core.config import get_settings
from core.logging import get_logger
from models.schemas import VideoIngestedEvent

logger = get_logger(__name__)
settings = get_settings()


class MessageQueuePublisher:
    """Manages a single AMQP connection and channel for publishing events."""

    def __init__(self) -> None:
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None

    async def startup(self) -> None:
        self._connection = await aio_pika.connect_robust(settings.AMQP_URL)
        self._channel = await self._connection.channel()
        self._exchange = await self._channel.declare_exchange(
            name=settings.AMQP_EXCHANGE,
            type=ExchangeType.TOPIC,
            durable=True,
        )
        logger.info("amqp_publisher_ready", exchange=settings.AMQP_EXCHANGE)

    async def shutdown(self) -> None:
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
        logger.info("amqp_publisher_closed")

    async def publish_video_ingested(self, event: VideoIngestedEvent) -> None:
        """Publish a VideoIngestedEvent with persistent delivery mode."""
        if self._exchange is None:
            raise RuntimeError("Publisher not started – call startup() first")

        payload = event.model_dump_json().encode()
        message = Message(
            body=payload,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            headers={"event_type": event.event_type},
        )

        await self._exchange.publish(
            message,
            routing_key=settings.AMQP_ROUTING_KEY_INGESTED,
        )
        logger.info(
            "event_published",
            event_type=event.event_type,
            video_id=event.video_id,
            routing_key=settings.AMQP_ROUTING_KEY_INGESTED,
        )

    async def publish_raw(self, routing_key: str, payload: dict) -> None:
        """Generic publish helper for arbitrary events."""
        if self._exchange is None:
            raise RuntimeError("Publisher not started – call startup() first")
        body = json.dumps(payload).encode()
        message = Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        await self._exchange.publish(message, routing_key=routing_key)


# Singleton instance
mq_publisher = MessageQueuePublisher()