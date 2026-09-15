"""send_sign_in_link: per-email cooldown -- slowapi's per-IP limiter on the
route alone is trivial to rotate past to email-bomb one target inbox. No
prior coverage existed for this file at all.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from app.core.cache import redis_client
from app.services import auth_links


def _unique_email():
    return f"test-{uuid.uuid4().hex}@example.com"


@pytest.fixture
def _cleanup_cooldown_key():
    keys = []
    yield keys
    for key in keys:
        redis_client.delete(key)


def test_send_sign_in_link_sends_on_first_call(monkeypatch, _cleanup_cooldown_key):
    email = _unique_email()
    _cleanup_cooldown_key.append(f"authmail:cooldown:{email}")

    monkeypatch.setattr(auth_links.firebase_auth, "generate_sign_in_with_email_link",
                         lambda email, settings: "https://sign-in.test/link")
    sent = MagicMock()
    monkeypatch.setattr(auth_links, "send_email", sent)

    auth_links.send_sign_in_link(email, "https://app.test/continue")

    sent.assert_called_once()


def test_send_sign_in_link_rejects_a_second_call_within_the_cooldown(monkeypatch, _cleanup_cooldown_key):
    """Real gap found live: only per-IP rate limiting guarded this endpoint,
    trivial to rotate past. A repeated request for the SAME email must be
    throttled regardless of source IP."""
    email = _unique_email()
    _cleanup_cooldown_key.append(f"authmail:cooldown:{email}")

    monkeypatch.setattr(auth_links.firebase_auth, "generate_sign_in_with_email_link",
                         lambda email, settings: "https://sign-in.test/link")
    sent = MagicMock()
    monkeypatch.setattr(auth_links, "send_email", sent)

    auth_links.send_sign_in_link(email, "https://app.test/continue")
    with pytest.raises(auth_links.SignInLinkRateLimitedError):
        auth_links.send_sign_in_link(email, "https://app.test/continue")

    sent.assert_called_once()  # not sent a second time


def test_send_sign_in_link_cooldown_is_per_email_not_global(monkeypatch, _cleanup_cooldown_key):
    email_a = _unique_email()
    email_b = _unique_email()
    _cleanup_cooldown_key.append(f"authmail:cooldown:{email_a}")
    _cleanup_cooldown_key.append(f"authmail:cooldown:{email_b}")

    monkeypatch.setattr(auth_links.firebase_auth, "generate_sign_in_with_email_link",
                         lambda email, settings: "https://sign-in.test/link")
    sent = MagicMock()
    monkeypatch.setattr(auth_links, "send_email", sent)

    auth_links.send_sign_in_link(email_a, "https://app.test/continue")
    auth_links.send_sign_in_link(email_b, "https://app.test/continue")  # different email -- must not be blocked

    assert sent.call_count == 2
