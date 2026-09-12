"""Price-tampering safety + correctness for subscription checkout & webhooks.

Same trust rules as the credit-pack flow:
  * the checkout route takes no request body — only `subscription_id` in the path
  * the plan_id (and therefore the amount Razorpay charges) comes only from the
    catalog row, never from the caller
  * handle_subscription_activated fetches the subscription server-to-server and
    never trusts the webhook payload's own notes
"""

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_db
from app.deps import get_current_user
from app.main import app
from app.services import billing as billing_svc
from app.services import webhooks as webhooks_svc

SUB_ID = uuid.uuid4()
TEAM_ID = uuid.uuid4()
REAL_PLAN_ID = "plan_real_monthly"


class FakePlan:
    id = SUB_ID
    slug = "monthly"
    period_label = "month"
    credits = 350
    razorpay_plan_id = REAL_PLAN_ID


class FakePlanUnconfigured(FakePlan):
    razorpay_plan_id = None


# --------------------------------------------------------------------------- #
# checkout endpoint
# --------------------------------------------------------------------------- #

def _url():
    return f"/billing/teams/{TEAM_ID}/subscriptions/{SUB_ID}/checkout"


@pytest.fixture
def sub_client(monkeypatch):
    fake_user = MagicMock(id=uuid.uuid4())
    state = {"plan": FakePlan(), "existing_team_sub": None, "created": []}

    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "Subscription":
            q.filter.return_value.first.return_value = state["plan"]
        else:  # TeamSubscription — checkout locks it with .with_for_update().first()
            q.filter.return_value.with_for_update.return_value.first.return_value = (
                state["existing_team_sub"]
            )
            q.filter.return_value.first.return_value = state["existing_team_sub"]
        return q

    db.query.side_effect = query

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: True)

    def fake_create(payload):
        state["created"].append(payload)
        return {"id": "sub_test_1"}

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "create", fake_create)

    yield TestClient(app, raise_server_exceptions=False), state, db
    app.dependency_overrides.clear()


def test_subscription_checkout_uses_db_plan_ignoring_injected_fields(sub_client):
    client, state, db = sub_client
    res = client.post(
        _url(),
        headers={"Authorization": "Bearer x"},
        json={"plan_id": "plan_ATTACKER", "amount": 1, "total_count": 999, "credits": 999999},
    )
    assert res.status_code == 200
    assert res.json()["razorpay_subscription_id"] == "sub_test_1"

    assert len(state["created"]) == 1
    sent = state["created"][0]
    assert sent["plan_id"] == REAL_PLAN_ID          # from DB, not "plan_ATTACKER"
    assert sent["total_count"] == 12                # month -> 12, not 999
    assert sent["notes"] == {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}
    assert "amount" not in sent
    assert 1 not in sent.values()


def test_subscription_checkout_blocks_when_active_row_exists(sub_client):
    client, state, db = sub_client
    state["existing_team_sub"] = MagicMock(status="active")
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400
    assert "already" in res.json()["detail"].lower()
    assert state["created"] == []


def test_subscription_checkout_blocks_when_pending_row_exists(sub_client):
    """The second of two rapid clicks: the first left a `pending` row behind."""
    client, state, db = sub_client
    state["existing_team_sub"] = MagicMock(status="pending")
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400
    assert state["created"] == []          # no second Razorpay subscription


def test_subscription_checkout_integrity_error_becomes_conflict(sub_client):
    """The DB-level race: two requests both pass the check, the loser's INSERT
    trips UNIQUE(team_id)."""
    from sqlalchemy.exc import IntegrityError

    client, state, db = sub_client
    state["existing_team_sub"] = None
    db.flush.side_effect = IntegrityError("dup", {}, Exception())
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400
    assert "already" in res.json()["detail"].lower()
    assert state["created"] == []
    db.rollback.assert_called()


def test_subscription_checkout_rolls_back_pending_on_razorpay_failure(sub_client, monkeypatch):
    """A Razorpay error must not leave a stuck `pending` row."""
    client, state, db = sub_client
    state["existing_team_sub"] = None

    def boom(payload):
        raise RuntimeError("razorpay down")

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "create", boom)
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 500
    db.rollback.assert_called()
    db.commit.assert_not_called()


def test_subscription_checkout_reuses_cancelled_row_on_resubscribe(sub_client):
    client, state, db = sub_client
    cancelled = MagicMock(status="cancelled")
    state["existing_team_sub"] = cancelled
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 200
    assert len(state["created"]) == 1          # a fresh Razorpay subscription
    assert cancelled.status == "pending"       # row reused, not a new insert
    assert cancelled.razorpay_subscription_id == "sub_test_1"  # real id stored, not None
    db.add.assert_not_called()
    db.commit.assert_called()


def test_two_rapid_checkouts_create_only_one_subscription(sub_client):
    """End-to-end: click 1 lands a pending row + one Razorpay subscription;
    click 2 sees that row and is rejected — no second subscription."""
    client, state, db = sub_client

    res1 = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res1.status_code == 200

    # click 1 left a pending row behind
    state["existing_team_sub"] = MagicMock(status="pending")

    res2 = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res2.status_code == 400
    assert len(state["created"]) == 1          # exactly one Razorpay subscription


def test_subscription_checkout_rejects_unconfigured_plan(sub_client):
    client, state, db = sub_client
    state["plan"] = FakePlanUnconfigured()
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400
    assert state["created"] == []


def test_subscription_checkout_requires_owner(sub_client, monkeypatch):
    client, state, db = sub_client
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: False)
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 403


# --------------------------------------------------------------------------- #
# subscription.activated webhook
# --------------------------------------------------------------------------- #

def _activated_db(plan, locked=None, team_sub_by_team=None):
    """`locked` = the row found (and locked) by razorpay_subscription_id — the
    normal case, since checkout now stores the id. `team_sub_by_team` is only
    consulted when `locked` is None (legacy / safety fallback)."""
    db = MagicMock()
    seen = {"ts": 0}

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            seen["ts"] += 1
            val = locked if seen["ts"] == 1 else team_sub_by_team
            q.filter.return_value.with_for_update.return_value.first.return_value = val
            q.filter.return_value.first.return_value = val
        elif name == "Subscription":
            q.filter.return_value.first.return_value = plan
        return q

    db.query.side_effect = query
    return db


def _sub_event(kind, sub_id="sub_test_1"):
    return {"event": kind, "payload": {"subscription": {"entity": {"id": sub_id}}}}


def test_activated_trusts_fetched_notes_not_webhook_payload(monkeypatch):
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda db, team_id, amount: refills.append((team_id, amount)),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "plan_id": REAL_PLAN_ID,
                     "notes": {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}},
    )
    db = _activated_db(FakePlan())

    # the webhook body itself carries nothing we trust — only the id is used
    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated"))

    assert refills == [(str(TEAM_ID), 350)]  # month plan -> full credits, fetched team
    db.add.assert_called_once()
    db.commit.assert_called()


def test_activated_rejects_plan_id_mismatch(monkeypatch):
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda *a, **k: refills.append(a),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "plan_id": "plan_TAMPERED",
                     "notes": {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}},
    )
    db = _activated_db(FakePlan())

    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated"))

    assert refills == []            # nothing granted
    db.add.assert_not_called()


def test_activated_is_idempotent_when_row_already_active(monkeypatch):
    fetch = MagicMock()
    monkeypatch.setattr(webhooks_svc.razorpay_client.subscription, "fetch", fetch)
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits", lambda *a, **k: None)

    active = MagicMock(status="active", razorpay_subscription_id="sub_new")
    db = _activated_db(FakePlan(), locked=active)
    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated", "sub_new"))

    fetch.assert_not_called()          # short-circuited before any work
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_activated_does_not_resurrect_a_cancelled_row(monkeypatch):
    """The row was cancelled (during the pending window). A late webhook for that
    same id must not reactivate it or grant credits."""
    refills = []
    fetch = MagicMock()
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits",
                        lambda *a, **k: refills.append(a))
    monkeypatch.setattr(webhooks_svc.razorpay_client.subscription, "fetch", fetch)

    cancelled = MagicMock(status="cancelled", razorpay_subscription_id="sub_new")
    db = _activated_db(FakePlan(), locked=cancelled)
    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated", "sub_new"))

    assert refills == []
    assert cancelled.status == "cancelled"   # unchanged
    fetch.assert_not_called()
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_activated_promotes_pending_row_from_checkout(monkeypatch):
    """The normal flow: checkout left a `pending` row carrying the real id;
    activation promotes it in place — no duplicate insert."""
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda db, team_id, amount: refills.append((team_id, amount)),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "plan_id": REAL_PLAN_ID,
                     "notes": {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}},
    )
    pending = MagicMock(status="pending", razorpay_subscription_id="sub_new")
    db = _activated_db(FakePlan(), locked=pending)

    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated", "sub_new"))

    db.add.assert_not_called()
    assert pending.status == "active"
    assert pending.razorpay_subscription_id == "sub_new"
    assert pending.credits_per_refill == 350
    assert refills == [(str(TEAM_ID), 350)]


def test_activated_inserts_when_id_not_on_any_row(monkeypatch):
    """Fallback: the id isn't stored anywhere (legacy / race) and the team has no
    row -> insert a fresh active row."""
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda db, team_id, amount: refills.append((team_id, amount)),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "plan_id": REAL_PLAN_ID,
                     "notes": {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}},
    )
    db = _activated_db(FakePlan(), locked=None, team_sub_by_team=None)

    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated", "sub_new"))

    db.add.assert_called_once()
    assert refills == [(str(TEAM_ID), 350)]


def test_activated_does_not_clobber_a_different_active_subscription(monkeypatch):
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits", lambda *a, **k: refills.append(a),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "plan_id": REAL_PLAN_ID,
                     "notes": {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}},
    )
    other = MagicMock(status="active", razorpay_subscription_id="sub_OTHER")
    db = _activated_db(FakePlan(), locked=None, team_sub_by_team=other)

    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated", "sub_new"))

    assert refills == []
    assert other.razorpay_subscription_id == "sub_OTHER"   # untouched
    db.commit.assert_not_called()


# --------------------------------------------------------------------------- #
# subscription.charged webhook
# --------------------------------------------------------------------------- #

def _charged_db(team_sub, plan):
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            q.filter.return_value.first.return_value = team_sub
            q.filter.return_value.with_for_update.return_value.first.return_value = team_sub
        elif name == "Subscription":
            q.filter.return_value.first.return_value = plan
        return q

    db.query.side_effect = query
    return db


def test_charged_first_payment_does_not_double_refill(monkeypatch):
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda *a, **k: refills.append(a),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 1},  # the initial charge
    )
    team_sub = MagicMock(
        team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
        credits_per_refill=350, last_paid_count=0,
    )
    db = _charged_db(team_sub, FakePlan())

    webhooks_svc.handle_subscription_charged(db, _sub_event("subscription.charged"))

    assert refills == []              # activation already granted this period
    assert team_sub.status == "active"


def test_charged_renewal_refills(monkeypatch):
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda db, team_id, amount: refills.append((team_id, amount)),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 2},  # a renewal
    )
    team_sub = MagicMock(
        team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
        credits_per_refill=350, last_paid_count=0,
    )
    db = _charged_db(team_sub, FakePlan())

    webhooks_svc.handle_subscription_charged(db, _sub_event("subscription.charged"))

    assert refills == [(str(TEAM_ID), 350)]


def test_charged_redelivered_webhook_does_not_double_refill_or_advance(monkeypatch):
    """Razorpay can redeliver the same subscription.charged event. The second
    delivery reports the SAME paid_count as the first, so it must be a no-op:
    refill_subscription_credits fires only once, and next_refill_at / the
    last_paid_count marker only advance once."""
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda db, team_id, amount: refills.append((team_id, amount)),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 2},  # a renewal, unchanged across both deliveries
    )
    team_sub = MagicMock(
        team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
        credits_per_refill=350, last_paid_count=1,
    )
    db = _charged_db(team_sub, FakePlan())
    event = _sub_event("subscription.charged")

    webhooks_svc.handle_subscription_charged(db, event)   # first delivery
    first_next_refill_at = team_sub.next_refill_at
    webhooks_svc.handle_subscription_charged(db, event)   # redelivered

    assert refills == [(str(TEAM_ID), 350)]        # only the first delivery refilled
    assert team_sub.last_paid_count == 2           # advanced once, not twice
    assert team_sub.next_refill_at == first_next_refill_at  # not pushed out again


def test_halted_and_cancelled_set_status(monkeypatch):
    for kind, expected in [("subscription.halted", "halted"), ("subscription.cancelled", "cancelled")]:
        team_sub = MagicMock()
        db = _charged_db(team_sub, FakePlan())
        handler = {
            "subscription.halted": webhooks_svc.handle_subscription_halted,
            "subscription.cancelled": webhooks_svc.handle_subscription_cancelled,
        }[kind]
        handler(db, _sub_event(kind))
        assert team_sub.status == expected
        db.commit.assert_called()
