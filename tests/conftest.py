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


class _BlockedSMTP:
    """Stand-in for smtplib.SMTP that refuses to connect and records the attempt."""
    attempts: list = []

    def __init__(self, host=None, port=None, *a, **k):
        _BlockedSMTP.attempts.append((host, port))
        raise ConnectionRefusedError(
            "the test suite attempted to open a REAL SMTP connection -- stub "
            "send_email (or the function that calls it) in this test"
        )


@pytest.fixture(autouse=True)
def _never_send_real_email(monkeypatch):
    """Hard guarantee that running the tests can never email a real person.

    Several tests run against the real dev database, whose users are real
    people with real inboxes, and the app sends genuine email through the
    configured SMTP account. One such test (a real-DB team delete/restore
    check) did exactly that -- every suite run emailed real users a
    "team restored" message. Stubbing send_email in each test is what should
    happen, but "every test remembers" is precisely how that leaked, so
    this enforces it: any attempt to open an SMTP connection is refused and
    then FAILS the test at teardown, even if the calling code swallows the
    error (notify_team_restored and friends deliberately do), pointing at
    the test that needs a stub.
    """
    from app.core import email as email_module

    _BlockedSMTP.attempts = []
    monkeypatch.setattr(email_module.smtplib, "SMTP", _BlockedSMTP)
    yield
    attempts, _BlockedSMTP.attempts = _BlockedSMTP.attempts, []
    assert not attempts, (
        f"this test tried to send real email ({len(attempts)} SMTP connection attempt(s) "
        f"to {attempts[0][0]}:{attempts[0][1]}) -- stub send_email in it"
    )
