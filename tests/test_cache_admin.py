from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)


def test_clear_cache_rejects_wrong_secret():
    res = client.post("/admin/cache/clear", headers={"x-cache-secret": "wrong"})
    assert res.status_code == 403


def test_clear_cache_requires_header():
    res = client.post("/admin/cache/clear")
    assert res.status_code == 422


def test_clear_cache_with_valid_secret():
    # warm the cache first
    client.get("/landing/tools")

    res = client.post(
        "/admin/cache/clear",
        headers={"x-cache-secret": settings.cache_clear_secret},
    )
    assert res.status_code == 200
    assert "cleared" in res.json()
