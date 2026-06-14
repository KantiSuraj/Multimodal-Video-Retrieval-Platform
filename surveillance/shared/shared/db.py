"""
Shared async SQLAlchemy engine + session factory.

Each service calls build_engine() with its own settings to get back
a correctly-configured engine.  The Base for table metadata lives in
shared.models so all services use the same metadata object.

Usage:
    from shared.db import build_engine, build_session_factory

    engine = build_engine(settings)
    AsyncSessionLocal = build_session_factory(engine)
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from shared.config.base import BaseServiceSettings


def build_engine(settings: BaseServiceSettings) -> AsyncEngine:
    return create_async_engine(
        settings.DATABASE_URL,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        echo=settings.DEBUG,
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )