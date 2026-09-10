"""Subscription cancellation + team-deletion-cancels-Razorpay."""

import logging
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_db
from app.deps import get_current_user
from app.main import app
from app.services import billing as billing_svc
from app.services import teams as teams_svc
from app.services import webhooks as webhooks_svc

TEAM_ID = uuid.uuid4()


# --------------------------------------------------------------------------- #
# cancel_subscription (service)
# --------------------------------------------------------------------------- #

def _cancel_db(team_sub):
    db = MagicMock()
    (db.query.return_value.filter.return_value
       .with_for_update.return_value.first.return_value) = team_sub
    return db


def test_cancel_no_subscription_raises_and_never_touches_razorpay(monkeypatch):
    calls = []
    monkeypatch.setattr(
        billing_svc.razorpay_client.subscription, "cancel", lambda sid: calls.append(sid)
    )
    db = _cancel_db(None)

    with pytest.raises(ValueError, match="no active subscription to cancel"):
        billing_svc.cancel_subscription(db, TEAM_ID)

    assert calls == []
    db.commit.assert_not_called()


def test_cancel_calls_razorpay_before_local_status_change(monkeypatch):
    team_sub = MagicMock(razorpay_subscription_id="sub_live_1", status="active")
    seen = {}

    def fake_cancel(sid):
        seen["sid"] = sid
        seen["status_at_call_time"] = team_sub.status  # must still be 'active'

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "cancel", fake_cancel)
    db = _cancel_db(team_sub)

    result = billing_svc.cancel_subscription(db, TEAM_ID)

    assert seen == {"sid": "sub_live_1", "status_at_call_time": "active"}
    assert team_sub.status == "cancelled"        # flipped only after Razorpay returned
    db.commit.assert_called_once()
    assert result is team_sub


def test_cancel_razorpay_failure_leaves_status_unchanged(monkeypatch):
    team_sub = MagicMock(razorpay_subscription_id="sub_live_1", status="active")

    def boom(sid):
        raise RuntimeError("razorpay 500")

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "cancel", boom)
    db = _cancel_db(team_sub)

    with pytest.raises(billing_svc.RazorpayCancelError):
        billing_svc.cancel_subscription(db, TEAM_ID)

    assert team_sub.status == "active"           # NOT falsely 'cancelled'
    db.commit.assert_not_called()


def test_cancel_pending_row_with_stored_id_still_calls_razorpay(monkeypatch):
    """Checkout now stores the real id on the pending row, so cancelling during
    the pending window DOES reach Razorpay."""
    calls = []
    monkeypatch.setattr(
        billing_svc.razorpay_client.subscription, "cancel", lambda sid: calls.append(sid)
    )
    team_sub = MagicMock(razorpay_subscription_id="sub_live_7", status="pending")
    db = _cancel_db(team_sub)

    billing_svc.cancel_subscription(db, TEAM_ID)

    assert calls == ["sub_live_7"]
    assert team_sub.status == "cancelled"


def test_pending_cancel_then_late_activated_does_not_resurrect(monkeypatch):
    """The full sequence: pending row with a real id -> owner cancels (Razorpay
    reached) -> a late subscription.activated for that id is a clean no-op."""
    SUB = "sub_live_42"
    row = MagicMock(status="pending", razorpay_subscription_id=SUB,
                    subscription_id=uuid.uuid4(), team_id=str(TEAM_ID))

    # cancel during the pending window
    cancel_calls = []
    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "cancel",
                        lambda sid: cancel_calls.append(sid))
    billing_svc.cancel_subscription(_cancel_db(row), TEAM_ID)
    assert cancel_calls == [SUB]          # id not null -> Razorpay actually told to stop
    assert row.status == "cancelled"

    # late-arriving subscription.activated for the same id
    refills = []
    fetch = MagicMock()
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits",
                        lambda *a, **k: refills.append(a))
    monkeypatch.setattr(webhooks_svc.razorpay_client.subscription, "fetch", fetch)

    wdb = MagicMock()
    (wdb.query.return_value.filter.return_value
       .with_for_update.return_value.first.return_value) = row
    webhooks_svc.handle_subscription_activated(
        wdb,
        {"event": "subscription.activated",
         "payload": {"subscription": {"entity": {"id": SUB}}}},
    )

    assert row.status == "cancelled"      # NOT resurrected
    assert refills == []                  # no credits granted
    fetch.assert_not_called()             # short-circuited before any work
    wdb.commit.assert_not_called()


# --------------------------------------------------------------------------- #
# POST /billing/teams/{team_id}/subscriptions/cancel (route)
# --------------------------------------------------------------------------- #

@pytest.fixture
def cancel_client(monkeypatch):
    fake_user = MagicMock(id=uuid.uuid4())
    state = {"team_sub": MagicMock(razorpay_subscription_id="sub_live_1", status="active"),
             "razorpay_ok": True}

    db = MagicMock()
    (db.query.return_value.filter.return_value
       .with_for_update.return_value.first.side_effect) = lambda: state["team_sub"]

    def cancel(sid):
        if not state["razorpay_ok"]:
            raise RuntimeError("razorpay down")

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "cancel", cancel)

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: True)

    yield TestClient(app, raise_server_exceptions=False), state
    app.dependency_overrides.clear()


def _url():
    return f"/billing/teams/{TEAM_ID}/subscriptions/cancel"


def test_cancel_route_requires_owner(cancel_client, monkeypatch):
    client, state = cancel_client
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: False)
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 403


def test_cancel_route_success(cancel_client):
    client, state = cancel_client
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 200
    assert res.json() == {"status": "cancelled", "teamId": str(TEAM_ID)}
    assert state["team_sub"].status == "cancelled"


def test_cancel_route_no_subscription_400(cancel_client):
    client, state = cancel_client
    state["team_sub"] = None
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400


def test_cancel_route_razorpay_failure_400(cancel_client):
    client, state = cancel_client
    state["razorpay_ok"] = False
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400
    assert "razorpay" in res.json()["detail"].lower()
    assert state["team_sub"].status == "active"      # unchanged


# --------------------------------------------------------------------------- #
# delete_team -> cancel_subscription
# --------------------------------------------------------------------------- #

def test_delete_team_cancels_subscription_first(monkeypatch):
    calls = []
    monkeypatch.setattr(teams_svc, "cancel_subscription", lambda db, tid: calls.append(tid))
    db = MagicMock()

    teams_svc.delete_team(db, TEAM_ID)

    assert calls == [TEAM_ID]
    db.commit.assert_called_once()


def test_delete_team_with_no_subscription_still_deletes(monkeypatch):
    def no_sub(db, tid):
        raise ValueError("This team has no active subscription to cancel")

    monkeypatch.setattr(teams_svc, "cancel_subscription", no_sub)
    db = MagicMock()

    teams_svc.delete_team(db, TEAM_ID)  # must not raise

    db.commit.assert_called_once()


def test_delete_team_razorpay_failure_still_deletes_and_logs(monkeypatch, caplog):
    def boom(db, tid):
        raise billing_svc.RazorpayCancelError("provider down")

    monkeypatch.setattr(teams_svc, "cancel_subscription", boom)
    db = MagicMock()

    with caplog.at_level(logging.ERROR, logger="app.services.teams"):
        teams_svc.delete_team(db, TEAM_ID)

    db.commit.assert_called_once()
    assert "MANUAL FOLLOW-UP NEEDED" in caplog.text
    # exc_info=True attaches the traceback
    assert any(r.exc_info for r in caplog.records)
