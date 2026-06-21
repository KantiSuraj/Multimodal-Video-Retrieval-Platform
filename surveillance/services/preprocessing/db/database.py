"""
services/preprocessing/db/database.py

Identical pattern to services/ingestion/db/database.py: module-level
singletons built from the shared factories. Preprocessing has no FastAPI
app and no Depends(get_db) — it's a pure worker, so sessions are opened
directly inside the orchestrator the same way ingestion's fs_watcher.py
opens its own session outside of HTTP context.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from shared.db import build_engine, build_session_factory

from services.preprocessing.core.config import get_settings

settings = get_settings()
engine = build_engine(settings)
AsyncSessionLocal = build_session_factory(engine)


@asynccontextmanager
async def get_session():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
