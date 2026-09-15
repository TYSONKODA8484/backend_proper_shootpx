import json
import logging

import redis

from app.core.config import settings

logger = logging.getLogger(__name__)

redis_client = redis.from_url(settings.redis_url, decode_responses=True)

CACHE_TTL_SECONDS = 3600  # 1 hour


def get_cached(key: str):
    """Return the cached value, or None. Never raises — cache is best-effort."""
    try:
        raw = redis_client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        logger.warning("cache get failed for %s", key, exc_info=True)
        return None


def set_cached(key: str, value, ttl: int = CACHE_TTL_SECONDS):
    """Store a JSON-serialisable value. Never raises — cache is best-effort."""
    try:
        redis_client.set(key, json.dumps(value, default=str), ex=ttl)
    except Exception:
        logger.warning("cache set failed for %s", key, exc_info=True)


def clear_cached(key: str):
    try:
        redis_client.delete(key)
    except Exception:
        logger.warning("cache clear failed for %s", key, exc_info=True)


def acquire_cooldown(key: str, ttl_seconds: int) -> bool:
    """
    Returns True if the cooldown was acquired (this key hasn't been used
    within ttl_seconds) -- False if it's still on cooldown. For throttling
    an action keyed by something other than the caller's IP -- e.g. a
    target email address, so an attacker can't email-bomb one victim by
    simply rotating IPs past slowapi's per-IP limiter.
    """
    return redis_client.set(key, "1", nx=True, ex=ttl_seconds) is not None
