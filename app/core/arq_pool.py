import asyncio

from arq import create_pool
from arq.connections import RedisSettings
from app.core.config import settings

_pool = None
_pool_lock = asyncio.Lock()


async def get_arq_pool():
    """
    Lazy singleton -- without the lock, two concurrent first-callers (e.g.
    two /generate requests racing at cold start) could each see _pool is None
    and each call create_pool(), leaking whichever pool's connection loses
    the race to be the one actually kept in _pool.
    """
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:  # re-check: another caller may have won the race while we waited
                _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return _pool