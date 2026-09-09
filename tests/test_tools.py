from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_tools_returns_list():
    res = client.get("/landing/tools")
    assert res.status_code == 200

    body = res.json()
    assert "tools" in body
    assert isinstance(body["tools"], list)


def test_tool_shape():
    res = client.get("/landing/tools")
    tools = res.json()["tools"]
    if tools:
        first = tools[0]
        for key in ("id", "slug", "name", "description", "category", "status", "sortOrder"):
            assert key in first
