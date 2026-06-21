"""
services/preprocessing/api/routes.py

Preprocessing's primary interface is RabbitMQ, not HTTP — there is no
upload endpoint, no client-facing contract. The only HTTP surface is a
liveness/readiness probe, which exists for the same reason every service
in this platform exposes one: Kubernetes needs something to poll.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
