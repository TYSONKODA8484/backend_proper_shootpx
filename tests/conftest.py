from unittest.mock import MagicMock

import pytest

from app.core.limiter import limiter
from app import worker
from app.services import generation as generation_svc
from app.services import tool_definitions as tool_definitions_svc


@pytest.fixture(autouse=True)
def _disable_rate_limiter():
    """Rate limiting uses a shared Redis store; disable it in tests so counters
    from one test (or a previous run) don't make another test flaky."""
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture(autouse=True)
def _mock_fal_slot_by_default(monkeypatch):
    """release_fal_slot / try_reserve_fal_slot talk to real Redis. Almost no
    test in this suite is actually exercising the fal-concurrency-limit
    feature itself, so auto-mock every caller's own imported reference to
    them by default -- otherwise any test that reaches a webhook, sweep, or
    submit code path corrupts the real dev Redis inflight counters with
    unmatched increments/decrements. Found live (twice): this exact gap
    drove the global counter to -14.

    A test that wants the real Redis-backed behavior (e.g.
    test_fal_slot_cap_real_redis_global_and_per_team) imports
    try_reserve_fal_slot/release_fal_slot directly from
    app.services.generation_lock and calls them directly, bypassing this.
    A test that wants to assert on a call (e.g. "released exactly once with
    this team_id") sets its own explicit monkeypatch, which simply overrides
    this default for that test."""
    monkeypatch.setattr(generation_svc, "release_fal_slot", MagicMock())
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())
    monkeypatch.setattr(worker, "try_reserve_fal_slot", MagicMock(return_value=True))


@pytest.fixture(autouse=True)
def _disable_tool_definition_cache(monkeypatch):
    """get_tool_definition() caches ToolDefinition rows in real Redis
    (60s TTL, key "tooldef:<feature_type>"). Almost every test in this suite
    mocks db.query(...) directly for a ToolDefinition lookup and expects that
    mock to be hit every time -- a real cache hit would skip the DB entirely
    and return whatever a PREVIOUS test (or real dev traffic) last cached
    under the same feature_type, and a real cache set would try to
    json.dumps() a MagicMock's attributes into actual dev Redis. Same class
    of bug already hit once with the fal-slot counters above -- force every
    lookup here to behave as a permanent cache miss with no writes."""
    monkeypatch.setattr(tool_definitions_svc, "get_cached", lambda key: None)
    monkeypatch.setattr(tool_definitions_svc, "set_cached", lambda key, value, ttl=None: None)
