"""app/services/signup_abuse.py -- caps the free-signup credit bonus at 5 per
rolling 24h window, counted by IP and device_id together via real Redis
INCR+EXPIRE (same pattern as generation_lock.py), plus a static disposable-
email-domain check. These tests use the real redis_client directly (same
style as test_tool_definitions.py's real_cache fixture) since the whole
point is exercising the actual counters, not mocking them away.
"""
import time
import uuid

import pytest

from app.core.cache import redis_client
from app.services import signup_abuse
from app.services.users import get_or_create_user


@pytest.fixture(autouse=True)
def _clean_signup_bonus_keys():
    yield
    for key in redis_client.keys("signup_bonus:*"):
        redis_client.delete(key)


@pytest.fixture(autouse=True)
def _clean_cached_test_users():
    """get_or_create_user now caches firebase_uid -> user in real Redis (a
    ~160ms DB round trip saved on every authenticated request). That cache
    outlives the test process, so a rerun with the same fake uids would be
    served a PREVIOUS run's cached user and skip account creation entirely.
    Real Firebase uids are globally unique and user rows are never deleted,
    so this is a test-isolation concern only -- but the fake uids below have
    to be cleaned up, and made unique per run (see _uid)."""
    yield
    for key in redis_client.keys("user:uid:test-*"):
        redis_client.delete(key)


def _uid(n: int) -> str:
    """Unique per test RUN, so a leftover cache entry from an earlier run can
    never satisfy this run's lookup."""
    return f"test-{uuid.uuid4().hex[:12]}-{n}"


def _unique_ip():
    return f"203.0.113.{uuid.uuid4().int % 250}"


def test_first_five_signups_from_same_ip_device_get_the_bonus():
    ip = _unique_ip()
    device_id = "device-abc"
    for _ in range(5):
        assert signup_abuse.should_grant_signup_bonus(ip, device_id) is True
        signup_abuse.record_signup_bonus_grant(ip, device_id)


def test_sixth_signup_from_same_ip_device_is_capped():
    ip = _unique_ip()
    device_id = "device-abc"
    for _ in range(5):
        signup_abuse.record_signup_bonus_grant(ip, device_id)

    assert signup_abuse.should_grant_signup_bonus(ip, device_id) is False


def test_cap_is_per_ip_and_per_device_independently():
    """Same device, different IP -- still capped, because EITHER counter
    being at/over 5 blocks the bonus (not just both)."""
    device_id = "device-shared"
    ip_a = _unique_ip()
    for _ in range(5):
        signup_abuse.record_signup_bonus_grant(ip_a, device_id)

    ip_b = _unique_ip()
    assert signup_abuse.should_grant_signup_bonus(ip_b, device_id) is False


def test_disposable_email_domain_is_flagged():
    assert signup_abuse.is_disposable_email("someone@mailinator.com") is True
    assert signup_abuse.is_disposable_email("someone@gmail.com") is False
    assert signup_abuse.is_disposable_email(None) is False


def test_no_device_id_only_checks_ip():
    ip = _unique_ip()
    for _ in range(5):
        signup_abuse.record_signup_bonus_grant(ip, None)

    assert signup_abuse.should_grant_signup_bonus(ip, None) is False


# --------------------------------------------------------------------------- #
# End-to-end through the real signup path (get_or_create_user)
# --------------------------------------------------------------------------- #

def test_sixth_rapid_signup_same_ip_and_device_gets_account_but_no_bonus(monkeypatch):
    """The actual scenario asked for: 6 rapid signups from the same IP+device.
    Every one gets a real account; only the first 5 get the starter credit."""
    from app.models.team import Team
    from app.models.team_member import TeamMember
    from app.models.user import User

    ip = _unique_ip()
    device_id = "device-e2e"

    created_teams = []

    class FakeDB:
        def __init__(self):
            self.added = []

        def add(self, obj):
            self.added.append(obj)
            if isinstance(obj, Team):
                created_teams.append(obj)

        def flush(self):
            for obj in self.added:
                if isinstance(obj, (User, Team)) and getattr(obj, "id", None) is None:
                    obj.id = uuid.uuid4()

        def commit(self):
            pass

        def rollback(self):
            pass

        def query(self, model):
            class _Q:
                def filter(self, *a, **k):
                    return self

                def first(self):
                    return None  # every call here is "brand new user"
            return _Q()

    # log_signup_bonus_attempt does a real DB insert -- irrelevant to this
    # test's assertion (bonus granted or not) and no real DB session exists
    # in this fake-db setup, so make it inert here.
    monkeypatch.setattr(signup_abuse, "log_signup_bonus_attempt", lambda *a, **k: None)

    class FakeBackgroundTasks:
        def add_task(self, fn, *args, **kwargs):
            fn(*args, **kwargs)

    for i in range(6):
        db = FakeDB()
        decoded = {"uid": _uid(i), "email": f"user{i}@example.com", "name": f"User {i}"}
        user = get_or_create_user(
            db, decoded, ip=ip, device_id=device_id, background_tasks=FakeBackgroundTasks(),
        )
        assert user is not None  # every one of the 6 gets a real account

    assert len(created_teams) == 6
    granted = [t.topup_credits_balance for t in created_teams]
    assert granted == [5, 5, 5, 5, 5, 0]  # only the first 5 get the starter bonus


def test_signup_bonus_check_is_fast():
    """2 Redis round-trips (GET, GET) -- not a DB query. Generous ceiling
    (50ms) just to catch a real regression (e.g. an accidental DB call),
    not to benchmark Redis itself."""
    ip = _unique_ip()
    start = time.perf_counter()
    signup_abuse.should_grant_signup_bonus(ip, "device-perf")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.05
