import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_db
from app.deps import get_current_user
from app.main import app
from app.models.tool_definition import ToolDefinition

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


# --------------------------------------------------------------------------- #
# GET /tools/{feature_type}/schema -- deliberately filtered subset of
# tool_definitions, used by clients to build the /generate form dynamically.
# --------------------------------------------------------------------------- #

@pytest.fixture
def schema_client():
    fake_user = MagicMock(id=uuid.uuid4())
    db = MagicMock()
    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app), db
    app.dependency_overrides.clear()


def test_tool_schema_returns_only_the_frontend_safe_fields(schema_client):
    """This is the real regression test for the bug found live during this
    audit: ToolDefinition.stage doesn't exist as an attribute error is raised
    -- a MagicMock stand-in for `db` does NOT hide this, because
    `ToolDefinition.stage == 1` is evaluated as a real expression against the
    actual SQLAlchemy model class before it ever reaches `db`."""
    client, db = schema_client
    tool = ToolDefinition(
        feature_type="recolor",
        fal_model_id="fal-ai/some-recolor-model",  # must NEVER reach the response
        max_input_images=3,
        max_output_resolution="2048x2048",
        default_output_count=1,
        credit_cost_per_output=5,
        param_schema=[{"name": "color", "type": "text", "required": True, "label": "Color"}],
        is_active=True,
        ai_steps={"detect_target": {"model": "fal-ai/moondream-next"}},  # must NEVER reach the response
        stage=1,
    )
    db.query.return_value.filter.return_value.first.return_value = tool

    res = client.get("/tools/recolor/schema", headers={"Authorization": "Bearer x"})

    assert res.status_code == 200
    body = res.json()
    assert body == {
        "featureType": "recolor",
        "maxInputImages": 3,
        "paramSchema": [{"name": "color", "type": "text", "required": True, "label": "Color"}],
    }
    assert "falModelId" not in body
    assert "aiSteps" not in body
    assert "fal_model_id" not in body
    assert "ai_steps" not in body


def test_tool_schema_404s_when_tool_missing(schema_client):
    client, db = schema_client
    db.query.return_value.filter.return_value.first.return_value = None

    res = client.get("/tools/ghost_tool/schema", headers={"Authorization": "Bearer x"})

    assert res.status_code == 404


def test_tool_schema_404s_when_tool_inactive(schema_client):
    client, db = schema_client
    tool = ToolDefinition(
        feature_type="recolor", fal_model_id="m", max_input_images=1,
        max_output_resolution="1024x1024", default_output_count=1,
        credit_cost_per_output=5, param_schema=[], is_active=False, ai_steps={}, stage=1,
    )
    db.query.return_value.filter.return_value.first.return_value = tool

    res = client.get("/tools/recolor/schema", headers={"Authorization": "Bearer x"})

    assert res.status_code == 404
