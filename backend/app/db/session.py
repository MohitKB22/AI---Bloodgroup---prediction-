"""FastAPI dependency for a per-request DB session, plus dev/test table
creation. Real deployments should manage schema via Alembic migrations
(backend/alembic/) rather than relying on create_all -- create_all is kept
here only as a zero-config convenience for local development and tests.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.base import Base, create_engine_and_sessionmaker

settings = get_settings()
engine, async_session_maker = create_engine_and_sessionmaker(settings.database_url)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


async def init_db() -> None:
    import app.models.prediction  # noqa: F401  (registers table with Base.metadata)
    import app.models.user  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
