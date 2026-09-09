from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.core.config import settings


def client_ip(request: Request) -> str:
    """The key rate limiting is bucketed by — the real client IP.

    In production the app sits behind a proxy / load balancer, so
    `request.client.host` is the proxy, not the user. When `TRUST_PROXY` is set we
    read the client IP from the end of `X-Forwarded-For` — that's the address the
    trusted proxy actually saw, which a client can't spoof past that proxy.

    Locally (`TRUST_PROXY=false`) we ignore the header, since anyone could send it.
    """
    if settings.trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[-1].strip()
    return get_remote_address(request)


# Counters live in Redis so the limit is shared across all workers / instances.
limiter = Limiter(
    key_func=client_ip,
    storage_uri=settings.redis_url,
)
