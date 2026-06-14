"""
Reusable AMQP consumer base class.

Services subclass BaseConsumer, override handle_message(), and call
start_consuming() in their worker entrypoint.

Example (preprocessing service):
    from shared.queue.consumer import BaseConsumer
    from shared.events import VideoIngestedEvent

    class PreprocessingConsumer(BaseConsumer):
        async def handle_message(self, body: bytes, routing_key: str) -> None:
            event = VideoIngestedEvent.model_validate_json(body)
            await self.preprocessor.process(event)
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import aio_pika
from aio_pika import ExchangeType

from shared.config.base import BaseServiceSettings


class BaseConsumer(ABC):
    """
    Manages one robust AMQP connection for consuming messages.
    Override handle_message() with your processing logic.
    """

    def __init__(
        self,
        settings:    BaseServiceSettings,
        queue_name:  str,
        routing_key: str,
        prefetch:    int = 10,
    ) -> None:
        self._settings    = settings
        self._queue_name  = queue_name
        self._routing_key = routing_key
        self._prefetch    = prefetch
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None

    async def start_consuming(self) -> None:
        """Connect and start consuming.  Runs until cancelled."""
        self._connection = await aio_pika.connect_robust(self._settings.AMQP_URL)
        async with self._connection:
            channel  = await self._connection.channel()
            await channel.set_qos(prefetch_count=self._prefetch)

            exchange = await channel.declare_exchange(
                name=self._settings.AMQP_EXCHANGE,
                type=ExchangeType.TOPIC,
                durable=True,
            )
            queue = await channel.declare_queue(self._queue_name, durable=True)
            await queue.bind(exchange, routing_key=self._routing_key)

            async with queue.iterator() as messages:
                async for message in messages:
                    async with message.process(requeue_on_timeout=True):
                        try:
                            await self.handle_message(message.body, message.routing_key or "")
                        except Exception as exc:
                            # Log and nack; message goes to dead-letter queue
                            await self._on_error(exc, message.body)
                            raise

    @abstractmethod
    async def handle_message(self, body: bytes, routing_key: str) -> None:
        """Override with message-processing logic."""
        ...

    async def _on_error(self, exc: Exception, body: bytes) -> None:
        """Called before nack.  Override to add alerting / DLQ forwarding."""
        pass

    async def stop(self) -> None:
        if self._connection and not self._connection.is_closed:
            await self._connection.close()