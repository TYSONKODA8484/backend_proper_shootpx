from arq import create_pool
from arq.connections import RedisSettings
from app.core.config import settings

_pool = None


async def get_arq_pool():
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return _pool