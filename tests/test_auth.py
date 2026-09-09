from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_me_requires_auth_header():
    res = client.get("/auth/me")
    assert res.status_code == 401


def test_me_rejects_non_bearer_scheme():
    res = client.get("/auth/me", headers={"Authorization": "Token abc"})
    assert res.status_code == 401


def test_me_rejects_empty_bearer():
    res = client.get("/auth/me", headers={"Authorization": "Bearer "})
    assert res.status_code == 401


def test_me_rejects_invalid_token():
    res = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert res.status_code == 401


def test_authmail_validates_body():
    # missing continue_url -> 422 before the service is called
    res = client.post("/auth/authmail", json={"email": "a@b.com"})
    assert res.status_code == 422
