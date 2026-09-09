import secrets

from fastapi import APIRouter, Header, HTTPException, Request
from redis.exceptions import RedisError

from app.core.cache import redis_client
from app.core.config import settings
from app.core.limiter import limiter

router = APIRouter(prefix="/admin", tags=["cache"])


@router.post("/cache/clear")
@limiter.limit("5/minute")
def clear_cache(request: Request, x_cache_secret: str = Header(...)):
    # constant-time compare so the secret can't be guessed by timing
    if not secrets.compare_digest(x_cache_secret, settings.cache_clear_secret):
        raise HTTPException(status_code=403, detail="Invalid secret")

    try:
        keys = redis_client.keys("landing:*")
        if keys:
            redis_client.delete(*keys)
    except RedisError:
        raise HTTPException(status_code=503, detail="Cache is unavailable")

    return {"cleared": len(keys)}
