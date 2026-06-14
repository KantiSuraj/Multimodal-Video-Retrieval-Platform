"""
Database engine and session factory for the ingestion service.

Builds on shared.db so the engine creation logic is not duplicated.
This module is the single place inside the ingestion service that
touches SQLAlchemy setup — everything else just calls get_db().
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator

from shared.shared.db import build_engine, build_session_factory
from services.ingestion.core.config import get_settings

settings = get_settings()

engine            = build_engine(settings)
AsyncSessionLocal = build_session_factory(engine)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a session and commits/rolls back automatically."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise