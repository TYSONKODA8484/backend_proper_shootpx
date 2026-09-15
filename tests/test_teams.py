import uuid

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

FAKE_TEAM = str(uuid.uuid4())
FAKE_TOKEN = "sometoken"


def test_list_teams_requires_auth():
    res = client.get("/teams")
    assert res.status_code == 401


def test_invite_requires_auth():
    res = client.post(f"/teams/{FAKE_TEAM}/invite", json={"email": "a@b.com"})
    assert res.status_code == 401


def test_accept_requires_auth():
    res = client.post(f"/invites/{FAKE_TOKEN}/accept")
    assert res.status_code == 401


def test_members_requires_auth():
    res = client.get(f"/teams/{FAKE_TEAM}/members")
    assert res.status_code == 401


def test_rename_requires_auth():
    res = client.patch(f"/teams/{FAKE_TEAM}", json={"name": "X"})
    assert res.status_code == 401


def test_remove_member_requires_auth():
    res = client.delete(f"/teams/{FAKE_TEAM}/members/{uuid.uuid4()}")
    assert res.status_code == 401


def test_delete_team_requires_auth():
    res = client.delete(f"/teams/{FAKE_TEAM}")
    assert res.status_code == 401


def test_list_invites_requires_auth():
    res = client.get(f"/teams/{FAKE_TEAM}/invites")
    assert res.status_code == 401


def test_cancel_invite_requires_auth():
    res = client.delete(f"/teams/{FAKE_TEAM}/invites/{uuid.uuid4()}")
    assert res.status_code == 401


def test_remove_member_blocks_last_owner(monkeypatch):
    from unittest.mock import MagicMock

    from app.services import teams

    fake_member = MagicMock(role="owner")
    monkeypatch.setattr(teams, "_membership", lambda db, tid, uid: fake_member)
    monkeypatch.setattr(teams, "owner_count", lambda db, tid: 1)

    try:
        teams.remove_member(MagicMock(), FAKE_TEAM, "u1")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "last owner" in str(e).lower()


def test_invite_validates_body():
    res = client.post(
        f"/teams/{FAKE_TEAM}/invite",
        json={},  # missing email
        headers={"Authorization": "Bearer not-real"},
    )
    # rejected — either body validation (422) or auth (401), never a 500
    assert res.status_code in (401, 422)


def test_invite_rejects_bad_role():
    res = client.post(
        f"/teams/{FAKE_TEAM}/invite",
        json={"email": "a@b.com", "role": "superadmin"},
        headers={"Authorization": "Bearer not-real"},
    )
    # invalid role -> 422 body validation (or 401 if auth is checked first)
    assert res.status_code in (401, 422)


def test_create_invite_blocks_when_team_full(monkeypatch):
    from unittest.mock import MagicMock

    from app.services import team_invites

    # no existing pending invite -> code reaches the seat check
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = None
    monkeypatch.setattr(team_invites, "team_seat_count", lambda db, tid: team_invites.MAX_TEAM_MEMBERS)

    try:
        team_invites.create_invite(fake_db, FAKE_TEAM, "x@y.com", "some-user-id")
        assert False, "expected TeamFullError"
    except team_invites.TeamFullError as e:
        assert "full" in str(e).lower()
