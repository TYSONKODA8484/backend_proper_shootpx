"""accept_invite: no prior coverage existed for this file. Focused on the
real race found live -- two concurrent accept-invite calls for the same user
both passing the "already a member?" check before either commits -- and the
basic happy/role-update paths around it.
"""

import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.cache import redis_client
from app.services import team_invites as team_invites_svc
from app.services.team_invites import InviteEmailRateLimitedError, TeamFullError


def _invite(status="pending", email="invitee@test.com", role="editor"):
    return MagicMock(
        id=uuid.uuid4(), team_id=uuid.uuid4(), email=email, role=role,
        status=status, token="tok-1",
    )


def _accept_db(invite, existing_member=None, member_count=1):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.side_effect = [invite, existing_member]
    db.query.return_value.filter.return_value.count.return_value = member_count
    return db


def test_accept_invite_adds_a_new_member_and_marks_accepted():
    invite = _invite()
    db = _accept_db(invite, existing_member=None, member_count=1)

    result = team_invites_svc.accept_invite(db, "tok-1", uuid.uuid4(), invite.email)

    db.add.assert_called_once()
    assert invite.status == "accepted"
    db.commit.assert_called_once()
    assert result is invite


def test_accept_invite_updates_role_for_an_existing_member():
    invite = _invite(role="owner")
    existing_member = MagicMock(role="editor")
    db = _accept_db(invite, existing_member=existing_member)

    team_invites_svc.accept_invite(db, "tok-1", uuid.uuid4(), invite.email)

    db.add.assert_not_called()  # already a member -- no new row
    assert existing_member.role == "owner"
    assert invite.status == "accepted"


def test_accept_invite_rejects_when_team_is_full():
    invite = _invite()
    db = _accept_db(invite, existing_member=None, member_count=team_invites_svc.MAX_TEAM_MEMBERS)

    with pytest.raises(TeamFullError):
        team_invites_svc.accept_invite(db, "tok-1", uuid.uuid4(), invite.email)


def test_accept_invite_handles_concurrent_accept_race_gracefully():
    """The real fix: two concurrent accept_invite() calls for the same user
    both see `member is None` before either commits. The DB's own unique
    constraint on (team_id, user_id) catches the second insert as an
    IntegrityError on flush -- it must be treated as an already-successful
    accept (the other request got there first), not surfaced as an error."""
    invite = _invite()
    db = _accept_db(invite, existing_member=None, member_count=1)
    db.flush.side_effect = IntegrityError("INSERT", {}, Exception("duplicate key"))

    result = team_invites_svc.accept_invite(db, "tok-1", uuid.uuid4(), invite.email)

    db.rollback.assert_called_once()
    assert invite.status == "accepted"  # still marked accepted, not left pending
    db.commit.assert_called_once()      # the function still completes normally
    assert result is invite


def test_accept_invite_rejects_wrong_token():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(ValueError, match="not found"):
        team_invites_svc.accept_invite(db, "bad-token", uuid.uuid4(), "x@test.com")


def test_accept_invite_rejects_already_used_invite():
    invite = _invite(status="accepted")
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = invite

    with pytest.raises(ValueError, match="already used"):
        team_invites_svc.accept_invite(db, "tok-1", uuid.uuid4(), invite.email)


# --------------------------------------------------------------------------- #
# cancel_invite / list_pending_invites: a mistyped or never-accepted invite
# previously burned a seat forever -- this frees it back up.
# --------------------------------------------------------------------------- #

def test_cancel_invite_marks_it_cancelled_and_frees_the_seat():
    team_id = uuid.uuid4()
    invite_id = uuid.uuid4()
    invite = MagicMock(id=invite_id, team_id=team_id, status="pending")
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = invite

    team_invites_svc.cancel_invite(db, team_id, invite_id)

    assert invite.status == "cancelled"
    db.commit.assert_called_once()


def test_cancel_invite_rejects_unknown_invite():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(ValueError, match="not found"):
        team_invites_svc.cancel_invite(db, uuid.uuid4(), uuid.uuid4())


def test_cancel_invite_rejects_an_already_accepted_invite():
    invite = MagicMock(status="accepted")
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = invite

    with pytest.raises(ValueError, match="no longer pending"):
        team_invites_svc.cancel_invite(db, uuid.uuid4(), uuid.uuid4())

    assert invite.status == "accepted"  # untouched


def test_list_pending_invites_returns_only_pending():
    team_id = uuid.uuid4()
    pending = MagicMock(id=uuid.uuid4(), email="a@test.com", role="editor", created_at=None)
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [pending]

    result = team_invites_svc.list_pending_invites(db, team_id)

    assert result == [{"id": str(pending.id), "email": "a@test.com", "role": "editor", "createdAt": None}]
    # the query itself must filter status == "pending" -- confirmed via the
    # actual filter args passed
    filter_args = db.query.return_value.filter.call_args.args
    assert any(
        getattr(getattr(a, "right", None), "value", None) == "pending" for a in filter_args
    )


def test_accept_invite_rejects_mismatched_email():
    invite = _invite(email="real@test.com")
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = invite

    with pytest.raises(ValueError, match="different email"):
        team_invites_svc.accept_invite(db, "tok-1", uuid.uuid4(), "attacker@test.com")


# --------------------------------------------------------------------------- #
# create_invite: per-target-email cooldown -- per-IP rate limiting on the
# route alone is trivial to rotate past to email-bomb one target inbox.
# --------------------------------------------------------------------------- #

def _create_invite_db(existing_invite=None, member_count=0, pending_count=0, team=None):
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamInvite":
            q.filter.return_value.first.return_value = existing_invite
            q.filter.return_value.count.return_value = pending_count
        elif name == "TeamMember":
            q.filter.return_value.count.return_value = member_count
        elif name == "Team":
            q.filter.return_value.first.return_value = team
        return q

    db.query.side_effect = query
    return db


@pytest.fixture
def _cleanup_invite_cooldown_key():
    keys = []
    yield keys
    for key in keys:
        redis_client.delete(key)


def test_create_invite_sends_on_first_call(monkeypatch, _cleanup_invite_cooldown_key):
    email = f"invitee-{uuid.uuid4().hex}@test.com"
    _cleanup_invite_cooldown_key.append(f"invite:cooldown:{email}")
    db = _create_invite_db(existing_invite=None, member_count=1, pending_count=0)

    monkeypatch.setattr(team_invites_svc.firebase_auth, "generate_sign_in_with_email_link",
                         lambda email, settings: "https://sign-in.test/link")
    sent = MagicMock()
    monkeypatch.setattr(team_invites_svc, "send_email", sent)

    team_invites_svc.create_invite(db, uuid.uuid4(), email, uuid.uuid4())

    sent.assert_called_once()


def test_create_invite_rejects_a_second_call_within_the_cooldown(monkeypatch, _cleanup_invite_cooldown_key):
    """Real gap found live: only per-IP rate limiting guarded this endpoint.
    Repeated invites to the SAME target email must be throttled regardless
    of source IP (an owner re-inviting rapidly, or abuse)."""
    email = f"invitee-{uuid.uuid4().hex}@test.com"
    _cleanup_invite_cooldown_key.append(f"invite:cooldown:{email}")
    # Second call: the invite now exists (created by the first call), so
    # the DB double is reconfigured to reflect that for realism.
    existing = MagicMock(role="editor", token="tok-1")
    db = _create_invite_db(existing_invite=None, member_count=1, pending_count=0)
    db_second = _create_invite_db(existing_invite=existing, member_count=1, pending_count=1)

    monkeypatch.setattr(team_invites_svc.firebase_auth, "generate_sign_in_with_email_link",
                         lambda email, settings: "https://sign-in.test/link")
    sent = MagicMock()
    monkeypatch.setattr(team_invites_svc, "send_email", sent)

    team_invites_svc.create_invite(db, uuid.uuid4(), email, uuid.uuid4())
    with pytest.raises(InviteEmailRateLimitedError):
        team_invites_svc.create_invite(db_second, uuid.uuid4(), email, uuid.uuid4())

    sent.assert_called_once()  # not sent a second time
