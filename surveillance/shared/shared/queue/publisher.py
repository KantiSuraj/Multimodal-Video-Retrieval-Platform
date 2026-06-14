"""
Reusable AMQP publisher base class.

Each service subclasses BasePublisher and adds typed publish_* methods.
The connection/channel lifecycle is handled here so services never
repeat the boilerplate.

Example (ingestion service):
    from shared.queue.publisher import BasePublisher
    from shared.events import VideoIngestedEvent

    class IngestionPublisher(BasePublisher):
        async def publish_video_ingested(self, event: VideoIngestedEvent) -> None:
            await self.publish(event, routing_key="video.ingested")
"""
from __future__ import annotations

import json

import aio_pika
from aio_pika import ExchangeType, Message
from pydantic import BaseModel

from shared.config.base import BaseServiceSettings


class BasePublisher:
    """
    Manages one robust AMQP connection + channel.
    Call startup() in app lifespan, shutdown() on teardown.
    """

    def __init__(self, settings: BaseServiceSettings) -> None:
        self._settings  = settings
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel:    aio_pika.abc.AbstractChannel          | None = None
        self._exchange:   aio_pika.abc.AbstractExchange          | None = None

    async def startup(self) -> None:
        self._connection = await aio_pika.connect_robust(self._settings.AMQP_URL)
        self._channel    = await self._connection.channel()
        self._exchange   = await self._channel.declare_exchange(
            name=self._settings.AMQP_EXCHANGE,
            type=ExchangeType.TOPIC,
            durable=True,
        )

    async def shutdown(self) -> None:
        if self._connection and not self._connection.is_closed:
            await self._connection.close()

    async def publish(self, event: BaseModel, *, routing_key: str) -> None:
        """Serialise a Pydantic event and publish with persistent delivery."""
        if self._exchange is None:
            raise RuntimeError("Publisher not started — call startup() first")
        body = event.model_dump_json().encode()
        await self._exchange.publish(
            Message(
                body=body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers={"event_type": getattr(event, "event_type", type(event).__name__)},
            ),
            routing_key=routing_key,
        )

    async def publish_raw(self, routing_key: str, payload: dict) -> None:
        """Publish an arbitrary dict — use only for one-off cases."""
        if self._exchange is None:
            raise RuntimeError("Publisher not started — call startup() first")
        await self._exchange.publish(
            Message(
                body=json.dumps(payload).encode(),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=routing_key,
        )