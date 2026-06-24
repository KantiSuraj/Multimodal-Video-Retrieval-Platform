"""Workers package for the Search Service.

The Search Service has no consumer worker.

Every other pipeline service (Embedding, Indexing) has a workers/
consumer_worker.py that bridges RabbitMQ to the service orchestrator.
Search is different: it is the terminal retrieval layer and is
request-driven through HTTP APIs, not event-driven through RabbitMQ.

This package exists to mirror the directory convention of the other
services.  No worker is registered here because there are no pipeline
events for Search to consume.

If a background task is ever added (e.g. scheduled cache warming,
index-health polling), it belongs here following the same
asyncio.Task + lifespan pattern used by the consumer workers.
"""
