"""Price-tampering safety for credit-pack checkout + webhook.

The client may only send `pack_id` (in the path). Neither the charge amount nor
the number of credits granted may ever come from client-supplied data — both are
derived server-side from the `credit` catalog row.
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

PACK_ID = uuid.uuid4()
TEAM_ID = uuid.uuid4()
PACK_PRICE = 50_000      # paise (₹500) — the real catalog price
PACK_CREDITS = 100       # the real catalog credit count


class FakePack:
    id = PACK_ID
    price = PACK_PRICE
    credits = PACK_CREDITS


# --------------------------------------------------------------------------- #
# checkout endpoint
# --------------------------------------------------------------------------- #

@pytest.fixture
def checkout_client(monkeypatch):
    """A signed-in owner, a mock DB that returns FakePack, and a captured
    razorpay order.create."""
    fake_user = MagicMock(id=uuid.uuid4())

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = FakePack()

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: True)

    created = []

    def fake_create(payload):
        created.append(payload)
        return {"id": "order_test_1", "amount": payload["amount"], "currency": payload["currency"]}

    monkeypatch.setattr(billing_svc.razorpay_client.order, "create", fake_create)

    yield TestClient(app), created
    app.dependency_overrides.clear()


def _url():
    return f"/billing/teams/{TEAM_ID}/credit-packs/{PACK_ID}/checkout"


def test_checkout_needs_no_body(checkout_client):
    client, created = checkout_client
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 200
    assert created[0]["amount"] == PACK_PRICE


def test_checkout_ignores_injected_amount_and_credits(checkout_client):
    client, created = checkout_client
    res = client.post(
        _url(),
        headers={"Authorization": "Bearer x"},
        json={
            "amount": 1,
            "price": 1,
            "credits": 999_999,
            "credits_added": 999_999,
            "pack": {"price": 1},
        },
    )
    assert res.status_code == 200
    body = res.json()

    # response amount is the DB price, not the injected 1
    assert body["amount"] == PACK_PRICE

    # the order Razorpay was told to create used the DB price and DB credits
    assert len(created) == 1
    sent = created[0]
    assert sent["amount"] == PACK_PRICE
    assert sent["currency"] == "INR"
    assert sent["notes"]["credits"] == str(PACK_CREDITS)
    assert sent["notes"]["credit_pack_id"] == str(PACK_ID)
    # nothing the client sent leaked into the order payload
    assert "price" not in sent
    assert 1 not in sent.values()


# --------------------------------------------------------------------------- #
# webhook
# --------------------------------------------------------------------------- #

class FakeTeam:
    def __init__(self, deleted_at=None):
        self.deleted_at = deleted_at


def _make_db(pack, existing_txn=None, team=None):
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "BillingTransaction":
            q.filter.return_value.first.return_value = existing_txn
        elif name == "Team":
            q.filter.return_value.first.return_value = team if team is not None else FakeTeam(deleted_at=None)
        else:  # Credit
            q.filter.return_value.first.return_value = pack
        return q

    db.query.side_effect = query
    return db


def _event(payment_notes, amount, order_id="order_test_1", payment_id="pay_test_1"):
    return {
        "event": "payment.captured",
        "payload": {"payment": {"entity": {
            "id": payment_id,
            "order_id": order_id,
            "amount": amount,
            "notes": payment_notes,
        }}},
    }


def test_webhook_grants_db_credits_ignoring_payment_notes(monkeypatch):
    granted = []
    monkeypatch.setattr(
        webhooks_svc, "add_topup_credits",
        lambda db, team_id, amount, commit=True: granted.append((team_id, amount)),
    )
    # the order WE created — authoritative team + pack id, correct amount
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.order, "fetch",
        lambda oid: {"amount": PACK_PRICE,
                     "notes": {"team_id": str(TEAM_ID), "credit_pack_id": str(PACK_ID)}},
    )
    db = _make_db(FakePack())

    # attacker-crafted payment notes: different team, inflated credits
    event = _event({"team_id": "attacker-team", "credits": "999999"}, amount=PACK_PRICE)
    webhooks_svc.handle_payment_captured(db, event)

    # credited the ORDER's team with the DB credit count — not the payment notes
    assert granted == [(str(TEAM_ID), PACK_CREDITS)]


def test_webhook_rejects_underpayment(monkeypatch):
    granted = []
    monkeypatch.setattr(
        webhooks_svc, "add_topup_credits",
        lambda *a, **k: granted.append(a),
    )
    # order says 100 paise, catalog pack is 50_000 — order was not ours
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.order, "fetch",
        lambda oid: {"amount": 100,
                     "notes": {"team_id": str(TEAM_ID), "credit_pack_id": str(PACK_ID)}},
    )
    db = _make_db(FakePack())

    event = _event({}, amount=100)
    webhooks_svc.handle_payment_captured(db, event)

    assert granted == []  # nothing granted on an amount mismatch


def test_webhook_holds_credits_for_a_soft_deleted_team(monkeypatch):
    """A real customer paid real money for this credit pack -- if the team is
    soft-deleted, the payment must not be silently swallowed OR silently
    credited to a team the owner can't currently see. It's held (a
    BillingTransaction row with 0 credits_added and a distinct status) for
    manual support review/refund, not auto-refunded on this function's own
    authority."""
    granted = []
    monkeypatch.setattr(
        webhooks_svc, "add_topup_credits",
        lambda *a, **k: granted.append(a),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.order, "fetch",
        lambda oid: {"amount": PACK_PRICE,
                     "notes": {"team_id": str(TEAM_ID), "credit_pack_id": str(PACK_ID)}},
    )
    from datetime import datetime, timezone
    deleted_team = FakeTeam(deleted_at=datetime.now(timezone.utc))
    db = _make_db(FakePack(), team=deleted_team)
    added_rows = []
    db.add.side_effect = added_rows.append

    event = _event({}, amount=PACK_PRICE)
    webhooks_svc.handle_payment_captured(db, event)

    assert granted == []  # credits NOT granted

    from app.models.billing_transaction import BillingTransaction
    txn = next(r for r in added_rows if isinstance(r, BillingTransaction))
    assert txn.status == "held_team_deleted"
    assert txn.credits_added == 0
    assert txn.amount == PACK_PRICE  # the real amount paid is still recorded, for the refund/review path
    db.commit.assert_called_once()


def test_webhook_grants_normally_for_a_non_deleted_team(monkeypatch):
    """Confirms the new deleted-team check doesn't false-positive on an
    ordinary, active team."""
    granted = []
    monkeypatch.setattr(
        webhooks_svc, "add_topup_credits",
        lambda db, team_id, amount, commit=True: granted.append((team_id, amount)),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.order, "fetch",
        lambda oid: {"amount": PACK_PRICE,
                     "notes": {"team_id": str(TEAM_ID), "credit_pack_id": str(PACK_ID)}},
    )
    db = _make_db(FakePack(), team=FakeTeam(deleted_at=None))

    webhooks_svc.handle_payment_captured(db, _event({}, amount=PACK_PRICE))

    assert granted == [(str(TEAM_ID), PACK_CREDITS)]


def test_webhook_is_idempotent(monkeypatch):
    granted = []
    monkeypatch.setattr(
        webhooks_svc, "add_topup_credits",
        lambda *a, **k: granted.append(a),
    )
    fetch = MagicMock()
    monkeypatch.setattr(webhooks_svc.razorpay_client.order, "fetch", fetch)

    db = _make_db(FakePack(), existing_txn=MagicMock())  # already processed
    webhooks_svc.handle_payment_captured(db, _event({}, amount=PACK_PRICE))

    assert granted == []
    fetch.assert_not_called()  # bailed out before doing any work
