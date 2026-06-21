
"""
Reusable AMQP consumer base class.

Services subclass BaseConsumer, override handle_message(), and call
start_consuming() in their worker entrypoint.

Failure handling:

    recoverable=True
        -> nack(requeue=True)

    recoverable=False
        -> nack(requeue=False)

Messages nacked with requeue=False are routed to the queue's
dead-letter exchange and become visible in the corresponding DLQ.

This implementation intentionally does NOT attempt to count retries.
RabbitMQ does not increment x-death for nack(requeue=True), so retry
counting requires a dedicated retry queue pattern (DLX + TTL queue),
which can be added later without changing service code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import aio_pika
from aio_pika import ExchangeType
import traceback
from shared.config.base import BaseServiceSettings


class BaseConsumer(ABC):
    """
    Reusable RabbitMQ consumer.

    Subclasses implement handle_message().
    """

    def __init__(
        self,
        settings: BaseServiceSettings,
        queue_name: str,
        routing_key: str,
        prefetch: int = 10,
    ) -> None:
        self._settings = settings
        self._queue_name = queue_name
        self._routing_key = routing_key
        self._prefetch = prefetch

        self._connection: (
            aio_pika.abc.AbstractRobustConnection | None
        ) = None

    async def start_consuming(self) -> None:
        """
        Connect to RabbitMQ and consume forever.

        Runs until cancelled.
        """

        self._connection = await aio_pika.connect_robust(
            self._settings.AMQP_URL
        )

        async with self._connection:
            channel = await self._connection.channel()

            await channel.set_qos(
                prefetch_count=self._prefetch
            )

            #
            # Main exchange
            #
            exchange = await channel.declare_exchange(
                name=self._settings.AMQP_EXCHANGE,
                type=ExchangeType.TOPIC,
                durable=True,
            )

            #
            # Dead-letter exchange
            #
            dlx_name = f"{self._settings.AMQP_EXCHANGE}.dlx"

            dlx = await channel.declare_exchange(
                name=dlx_name,
                type=ExchangeType.TOPIC,
                durable=True,
            )

            #
            # Dead-letter queue
            #
            dlq = await channel.declare_queue(
                f"{self._queue_name}.dlq",
                durable=True,
            )

            await dlq.bind(
                dlx,
                routing_key="#",
            )

            #
            # Main queue
            #
            queue = await channel.declare_queue(
                self._queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": dlx_name,
                },
            )

            await queue.bind(
                exchange,
                routing_key=self._routing_key,
            )

            async with queue.iterator() as messages:
                async for message in messages:
                    await self._handle_one(message)

    async def _handle_one(
        self,
        message: aio_pika.abc.AbstractIncomingMessage,
    ) -> None:
        """
        Process a single RabbitMQ delivery.
        """

        try:
            await self.handle_message(
                message.body,
                message.routing_key or "",
            )

        except Exception as exc:
            await self._on_error(
                exc,
                message.body,
            )

            recoverable = getattr(
                exc,
                "recoverable",
                False,
            )

            #
            # Recoverable failures:
            # requeue for another attempt.
            #
            if recoverable:
                await message.nack(
                    requeue=True,
                )
                return

            #
            # Permanent failures:
            # route to DLQ.
            #
            await message.nack(
                requeue=False,
            )
            return

        #
        # Success
        #
        await message.ack()

    @abstractmethod
    async def handle_message(
        self,
        body: bytes,
        routing_key: str,
    ) -> None:
        """
        Process a single message.

        Raise an exception on failure.

        Optional convention:

            raise SomeError(..., recoverable=True)

        to indicate transient failures.
        """
        ...

    async def _on_error(
        self,
        exc: Exception,
        body: bytes,
    ) -> None:
        print("\n========== PREPROCESSING FAILURE ==========")
        print("Exception:", repr(exc))
        traceback.print_exc()
        print("Message body:", body.decode(errors="ignore"))
        print("===========================================\n")
        # """
        # Hook for logging, metrics, alerting, etc.
        # """
        # pass

    async def stop(self) -> None:
        """
        Close AMQP connection.
        """
        if (
            self._connection
            and not self._connection.is_closed
        ):
            await self._connection.close()