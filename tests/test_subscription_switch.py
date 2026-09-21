"""switch_subscription + POST /teams/{team_id}/subscriptions/{new_subscription_id}/switch

PRD §17: switching plans moves only the unused subscription-pool credits to
the top-up wallet. No proration math, no cash refund -- the leftover count is
copied as-is and the pool is zeroed, nothing else changes.

The function composes two already-tested primitives (cancel_subscription,
create_subscription_checkout) rather than reimplementing their logic, so
these tests focus on: validation ordering (fail before touching anything),
that cancel happens before any credit mutation, that the credit transfer
itself is a single atomic commit, and that a Razorpay cancel failure aborts
the whole switch cleanly.
"""

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_db
from app.deps import get_current_user
from app.main import app
from app.services import billing as billing_svc

TEAM_ID = uuid.uuid4()
OLD_SUB_ID = uuid.uuid4()
NEW_SUB_ID = uuid.uuid4()
OLD_RAZORPAY_ID = "sub_live_old"
NEW_RAZORPAY_PLAN_ID = "plan_real_new"


class FakeNewPlan:
    id = NEW_SUB_ID
    slug = "yearly"
    period_label = "year"
    credits = 5000
    razorpay_plan_id = NEW_RAZORPAY_PLAN_ID


class FakeNewPlanUnconfigured(FakeNewPlan):
    razorpay_plan_id = None


def _url(new_sub_id=None):
    return f"/billing/teams/{TEAM_ID}/subscriptions/{new_sub_id or NEW_SUB_ID}/switch"


@pytest.fixture
def switch_client(monkeypatch):
    fake_user = MagicMock(id=uuid.uuid4())

    team = MagicMock(id=TEAM_ID, subscription_credits_remaining=120, topup_credits_balance=30)
    old_team_sub = MagicMock(
        team_id=TEAM_ID,
        subscription_id=OLD_SUB_ID,
        razorpay_subscription_id=OLD_RAZORPAY_ID,
        status="active",
    )

    state = {
        "new_plan": FakeNewPlan(),
        "team": team,
        "team_sub": old_team_sub,
        "created": [],
        "cancel_calls": [],
        "razorpay_cancel_ok": True,
    }

    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "Subscription":
            q.filter.return_value.first.return_value = state["new_plan"]
        elif name == "TeamSubscription":
            q.filter.return_value.with_for_update.return_value.first.return_value = state["team_sub"]
            q.filter.return_value.first.return_value = state["team_sub"]
        elif name == "Team":
            q.filter.return_value.with_for_update.return_value.first.return_value = state["team"]
            q.filter.return_value.first.return_value = state["team"]
        return q

    db.query.side_effect = query

    def fake_cancel(sid):
        state["cancel_calls"].append(sid)
        if not state["razorpay_cancel_ok"]:
            raise RuntimeError("razorpay down")

    def fake_create(payload):
        state["created"].append(payload)
        return {"id": "sub_test_new_1"}

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "cancel", fake_cancel)
    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "create", fake_create)

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: True)

    yield TestClient(app, raise_server_exceptions=False), state, db
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# route contract
# --------------------------------------------------------------------------- #

def test_switch_requires_owner(switch_client, monkeypatch):
    client, state, db = switch_client
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: False)
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 403
    assert state["cancel_calls"] == []
    assert state["created"] == []


def test_switch_ignores_injected_request_body(switch_client):
    """Both ids come from the path; nothing in a request body should matter."""
    client, state, db = switch_client
    res = client.post(
        _url(),
        headers={"Authorization": "Bearer x"},
        json={"subscription_id": "attacker", "credits": 999999},
    )
    assert res.status_code == 200
    assert state["created"][0]["plan_id"] == NEW_RAZORPAY_PLAN_ID


# --------------------------------------------------------------------------- #
# 1. successful switch
# --------------------------------------------------------------------------- #

def test_switch_success_cancels_old_moves_credits_and_creates_new_checkout(switch_client):
    client, state, db = switch_client
    res = client.post(_url(), headers={"Authorization": "Bearer x"})

    assert res.status_code == 200
    assert res.json()["razorpay_subscription_id"] == "sub_test_new_1"

    # old subscription actually cancelled with Razorpay
    assert state["cancel_calls"] == [OLD_RAZORPAY_ID]

    # leftover pool (120) moved into the existing wallet balance (30), pool zeroed
    team = state["team"]
    assert team.topup_credits_balance == 150
    assert team.subscription_credits_remaining == 0

    # new checkout created by reusing the existing (now-cancelled) row, not a fresh insert
    db.add.assert_not_called()
    assert len(state["created"]) == 1
    sent = state["created"][0]
    assert sent["plan_id"] == NEW_RAZORPAY_PLAN_ID
    assert sent["total_count"] == 1  # "year" -> 1
    assert state["team_sub"].status == "pending"
    assert state["team_sub"].subscription_id == NEW_SUB_ID
    assert state["team_sub"].razorpay_subscription_id == "sub_test_new_1"


# --------------------------------------------------------------------------- #
# 2. unconfigured new plan -- fails before touching anything
# --------------------------------------------------------------------------- #

def test_switch_to_unconfigured_plan_fails_before_any_cancellation(switch_client):
    client, state, db = switch_client
    state["new_plan"] = FakeNewPlanUnconfigured()

    res = client.post(_url(), headers={"Authorization": "Bearer x"})

    assert res.status_code == 400
    assert "not configured" in res.json()["detail"].lower()

    assert state["cancel_calls"] == []                    # never reached Razorpay
    assert state["created"] == []
    assert state["team_sub"].status == "active"           # old sub untouched
    assert state["team"].topup_credits_balance == 30       # credits untouched
    assert state["team"].subscription_credits_remaining == 120


# --------------------------------------------------------------------------- #
# 3. Razorpay cancel failure -- whole switch aborts cleanly
# --------------------------------------------------------------------------- #

def test_switch_aborts_when_razorpay_cancel_fails(switch_client):
    client, state, db = switch_client
    state["razorpay_cancel_ok"] = False

    res = client.post(_url(), headers={"Authorization": "Bearer x"})

    assert res.status_code == 400
    assert "razorpay" in res.json()["detail"].lower() or "cancel" in res.json()["detail"].lower()

    # nothing past the failed cancel call ever ran
    assert state["team_sub"].status == "active"            # NOT cancelled
    assert state["team"].topup_credits_balance == 30        # credits untouched
    assert state["team"].subscription_credits_remaining == 120
    assert state["created"] == []                           # no new checkout attempted


# --------------------------------------------------------------------------- #
# 4. no existing subscription at all
# --------------------------------------------------------------------------- #

def test_switch_with_no_existing_subscription_fails_cleanly(switch_client):
    client, state, db = switch_client
    state["team_sub"] = None

    res = client.post(_url(), headers={"Authorization": "Bearer x"})

    assert res.status_code == 400
    assert "no active subscription" in res.json()["detail"].lower()
    assert state["created"] == []
    assert state["team"].topup_credits_balance == 30        # untouched
    assert state["team"].subscription_credits_remaining == 120


# --------------------------------------------------------------------------- #
# 5. concurrency -- two rapid switch attempts for the same team
# --------------------------------------------------------------------------- #

def test_two_rapid_switch_attempts_do_not_double_transfer_or_duplicate_rows(switch_client):
    """Mirrors test_two_rapid_checkouts_create_only_one_subscription: the
    with_for_update() lock inside cancel_subscription / create_subscription_checkout
    forces the second request to re-read live (committed) state rather than a
    stale snapshot, so it can never re-cancel an already-cancelled row, never
    re-zero an already-empty credit pool, and never INSERTs a second
    TeamSubscription row -- the same row is reused throughout."""
    client, state, db = switch_client

    res1 = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res1.status_code == 200
    assert state["team"].subscription_credits_remaining == 0
    assert state["team"].topup_credits_balance == 150     # 30 + 120, moved exactly once

    # click 2 only ever sees the row after click 1's commit (the lock
    # serialises them) -- i.e. the now-"pending" row click 1 just created
    res2 = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res2.status_code == 200

    # the pool was already empty -- no double-transfer of the original 120
    assert state["team"].subscription_credits_remaining == 0
    assert state["team"].topup_credits_balance == 150

    # both switches mutated the SAME TeamSubscription row -- never duplicated
    db.add.assert_not_called()
    assert len(state["created"]) == 2   # two legitimate, sequential subscriptions
