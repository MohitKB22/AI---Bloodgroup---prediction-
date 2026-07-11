"""Fixed-window rate limiting, keyed per-user, backed by CacheService (Redis
or its in-memory fallback). Simple by design: a fixed window has a
well-known edge case (bursts across the window boundary), which is an
acceptable trade-off against a sliding-window/token-bucket implementation
for an endpoint whose real cost driver is GPU/CPU inference time, not
request counting precision.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status

from app.api.deps import get_cache_service, get_current_user
from app.core.config import get_settings
from app.models.user import User
from app.services.cache_service import CacheService


async def enforce_rate_limit(
    current_user: User = Depends(get_current_user),
    cache: CacheService = Depends(get_cache_service),
) -> None:
    settings = get_settings()
    window_seconds = 60
    key = f"ratelimit:predict:{current_user.id}:{_current_window(window_seconds)}"

    count = await cache.get_json(key) or 0
    if count >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: max {settings.rate_limit_per_minute} predictions per minute.",
        )
    await cache.set_json(key, count + 1, ttl_seconds=window_seconds)


def _current_window(window_seconds: int) -> int:
    import time
    return int(time.time() // window_seconds)
