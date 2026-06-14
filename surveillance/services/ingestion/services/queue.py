"""
Ingestion-specific AMQP publisher.

Subclasses shared.queue.BasePublisher and adds one typed publish method.
All connection management is inherited — this file is intentionally short.
"""
from __future__ import annotations

from shared.events.video_ingested import VideoIngestedEvent
from shared.queue.publisher import BasePublisher
from services.ingestion.core.config import get_settings

settings = get_settings()


class IngestionPublisher(BasePublisher):

    async def publish_video_ingested(self, event: VideoIngestedEvent) -> None:
        await self.publish(event, routing_key=settings.AMQP_ROUTING_KEY_INGESTED)


# Singleton — initialised in app lifespan
mq_publisher = IngestionPublisher(settings)