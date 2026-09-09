import uvicorn

from app.core.config import settings

if __name__ == "__main__":
    # Auto-reload only in development. In production the cloud/host runs the app
    # (e.g. `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 4`).
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,
        # when behind a proxy, let uvicorn rewrite request.client.host from
        # X-Forwarded-For (so logs + rate limiting see the real client)
        proxy_headers=settings.trust_proxy,
        forwarded_allow_ips="*" if settings.trust_proxy else None,
    )
