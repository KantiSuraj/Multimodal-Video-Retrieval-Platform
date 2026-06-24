"""Identical pattern to embedding/db/database.py — no HTTP request
boundary, so get_session() is an explicit async context manager called
directly from the orchestrator rather than a FastAPI dependency.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from shared.shared.db import build_engine, build_session_factory

from services.indexing.core.config import get_settings

settings = get_settings()
engine = build_engine(settings)
AsyncSessionLocal = build_session_factory(engine)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
