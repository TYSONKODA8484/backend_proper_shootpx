"""arq refill / cleanup jobs + the subscription.pending webhook."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from app import worker
from app.services import webhooks as webhooks_svc


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# refill_due_subscriptions
# --------------------------------------------------------------------------- #

def _refill_db(due=(), plan=None, lapse_due=(), team_for_update=None):
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            q.filter.return_value.all.return_value = list(due)              # active pass
            q.join.return_value.filter.return_value.all.return_value = list(lapse_due)  # lapse pass
        elif name == "Subscription":
            q.filter.return_value.first.return_value = plan
        elif name == "Team":
            (q.filter.return_value.with_for_update.return_value
               .first.return_value) = team_for_update
        return q

    db.query.side_effect = query
    return db


def test_refill_refills_and_advances_from_now(monkeypatch):
    calls = []
    monkeypatch.setattr(
        worker, "refill_subscription_credits",
        lambda db, team_id, amount, commit=True: calls.append((team_id, amount, commit)),
    )
    stale = datetime(2020, 1, 1, tzinfo=timezone.utc)   # years overdue
    ts = MagicMock(team_id="team-1", subscription_id="sub-1",
                   credits_per_refill=1000, next_refill_at=stale)
    db = _refill_db([ts], MagicMock(period_label="year"))
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.refill_due_subscriptions({}))

    # refill called once, as part of the caller's transaction (commit=False)
    assert calls == [("team-1", 1000, False)]
    # next_refill_at advanced from NOW (+~30d), NOT from 2020 -> no catch-up burst
    now = datetime.now(timezone.utc)
    assert now < ts.next_refill_at <= now + timedelta(days=31)
    db.commit.assert_called_once()
    db.close.assert_called_once()


def test_refill_step_per_period(monkeypatch):
    monkeypatch.setattr(worker, "refill_subscription_credits", lambda *a, **k: None)
    for label, lo, hi in [("week", 6, 8), ("month", 29, 31), ("year", 29, 31)]:
        ts = MagicMock(team_id="t", subscription_id="s", credits_per_refill=10,
                       next_refill_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
        db = _refill_db([ts], MagicMock(period_label=label))
        monkeypatch.setattr(worker, "SessionLocal", lambda: db)
        _run(worker.refill_due_subscriptions({}))
        delta = ts.next_refill_at - datetime.now(timezone.utc)
        assert timedelta(days=lo) <= delta <= timedelta(days=hi), label


def test_refill_one_failure_does_not_abort_the_batch(monkeypatch):
    def refill(db, team_id, amount, commit=True):
        if team_id == "bad":
            raise ValueError("Team not found")

    monkeypatch.setattr(worker, "refill_subscription_credits", refill)
    bad = MagicMock(team_id="bad", subscription_id="s", credits_per_refill=10,
                    next_refill_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    good = MagicMock(team_id="good", subscription_id="s", credits_per_refill=10,
                     next_refill_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    db = _refill_db([bad, good], MagicMock(period_label="week"))
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.refill_due_subscriptions({}))

    db.rollback.assert_called()                 # the bad one was rolled back
    assert isinstance(good.next_refill_at, datetime)   # the good one still processed
    db.close.assert_called_once()               # session always closed


def test_refill_skips_when_plan_missing(monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "refill_subscription_credits",
                        lambda *a, **k: calls.append(a))
    ts = MagicMock(team_id="t", subscription_id="s", credits_per_refill=10,
                   next_refill_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    db = _refill_db([ts], None)                  # plan lookup returns None
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.refill_due_subscriptions({}))

    assert calls == []
    db.close.assert_called_once()


# --------------------------------------------------------------------------- #
# refill_due_subscriptions — lapsing cancelled-subscription credits (PRD §5)
# --------------------------------------------------------------------------- #

def test_lapse_zeros_cancelled_pool_but_not_topup(monkeypatch):
    monkeypatch.setattr(worker, "refill_subscription_credits", lambda *a, **k: None)
    team = MagicMock(subscription_credits_remaining=70, topup_credits_balance=200)
    cancelled_sub = MagicMock(team_id="team-x")
    db = _refill_db(lapse_due=[cancelled_sub], team_for_update=team)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.refill_due_subscriptions({}))

    assert team.subscription_credits_remaining == 0
    assert team.topup_credits_balance == 200          # never touched
    db.commit.assert_called()
    db.close.assert_called_once()


def test_lapse_query_targets_cancelled_past_reset_with_credits(monkeypatch):
    """The 'not yet past reset' and 'active row' exclusions live in this filter:
    status == cancelled, next_refill_at <= now, subscription_credits_remaining > 0."""
    monkeypatch.setattr(worker, "refill_subscription_credits", lambda *a, **k: None)
    captured = {}
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            q.filter.return_value.all.return_value = []          # active pass: nothing

            def jf(*args):
                captured["args"] = args
                m = MagicMock()
                m.all.return_value = []
                return m

            q.join.return_value.filter.side_effect = jf
        return q

    db.query.side_effect = query
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.refill_due_subscriptions({}))

    sql = " ".join(str(a) for a in captured["args"])
    assert "team_subscriptions.status =" in sql
    assert "team_subscriptions.next_refill_at <=" in sql
    assert "teams.subscription_credits_remaining >" in sql


def test_lapse_one_failure_does_not_abort_batch(monkeypatch):
    monkeypatch.setattr(worker, "refill_subscription_credits", lambda *a, **k: None)
    s1, s2 = MagicMock(team_id="t1"), MagicMock(team_id="t2")
    team = MagicMock(subscription_credits_remaining=10)
    db = _refill_db(lapse_due=[s1, s2], team_for_update=team)
    db.commit.side_effect = [RuntimeError("boom"), None]   # 1st lapse commit fails
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.refill_due_subscriptions({}))

    db.rollback.assert_called()               # failed one rolled back
    assert db.commit.call_count == 2          # still attempted the second
    db.close.assert_called_once()


def test_lapse_pass_is_a_noop_when_nothing_cancelled(monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "refill_subscription_credits",
                        lambda *a, **k: calls.append(a))
    db = _refill_db()                          # no active due, no lapse due
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.refill_due_subscriptions({}))

    assert calls == []
    db.commit.assert_not_called()
    db.close.assert_called_once()


# --------------------------------------------------------------------------- #
# cleanup_stale_pending_subscriptions
# --------------------------------------------------------------------------- #

def test_cleanup_only_filters_pending_rows_older_than_30_min(monkeypatch):
    captured = {}
    db = MagicMock()

    def query(model):
        q = MagicMock()

        def filt(*args):
            captured["args"] = args
            m = MagicMock()
            m.all.return_value = []
            return m

        q.filter.side_effect = filt
        return q

    db.query.side_effect = query
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.cleanup_stale_pending_subscriptions({}))

    sql = " ".join(str(a) for a in captured["args"])
    # status == 'pending'  (so active / cancelled / halted are excluded outright)
    assert "team_subscriptions.status =" in sql
    # created_at <= cutoff  (so fresh pending rows are excluded)
    assert "team_subscriptions.created_at <=" in sql
    db.close.assert_called_once()


def test_cleanup_deletes_the_returned_stale_rows(monkeypatch):
    r1 = MagicMock(team_id="a")
    r2 = MagicMock(team_id="b")
    db = MagicMock()

    def query(model):
        q = MagicMock()
        q.filter.return_value.all.return_value = [r1, r2]
        return q

    db.query.side_effect = query
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.cleanup_stale_pending_subscriptions({}))

    db.delete.assert_any_call(r1)
    db.delete.assert_any_call(r2)
    db.commit.assert_called_once()
    db.close.assert_called_once()


# --------------------------------------------------------------------------- #
# handle_subscription_pending
# --------------------------------------------------------------------------- #

def _pending_event(sub_id="sub_x"):
    return {"event": "subscription.pending",
            "payload": {"subscription": {"entity": {"id": sub_id}}}}


def _pending_db(team_sub):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = team_sub
    return db


def test_pending_flips_active_to_pending():
    ts = MagicMock(status="active")
    db = _pending_db(ts)
    webhooks_svc.handle_subscription_pending(db, _pending_event())
    assert ts.status == "pending"
    db.commit.assert_called_once()


def test_pending_does_not_touch_cancelled():
    ts = MagicMock(status="cancelled")
    db = _pending_db(ts)
    webhooks_svc.handle_subscription_pending(db, _pending_event())
    assert ts.status == "cancelled"
    db.commit.assert_not_called()


def test_pending_does_not_touch_halted():
    ts = MagicMock(status="halted")
    db = _pending_db(ts)
    webhooks_svc.handle_subscription_pending(db, _pending_event())
    assert ts.status == "halted"
    db.commit.assert_not_called()


def test_pending_does_not_touch_already_pending():
    ts = MagicMock(status="pending")
    db = _pending_db(ts)
    webhooks_svc.handle_subscription_pending(db, _pending_event())
    assert ts.status == "pending"
    db.commit.assert_not_called()


def test_pending_noop_when_no_row():
    db = _pending_db(None)
    webhooks_svc.handle_subscription_pending(db, _pending_event())
    db.commit.assert_not_called()
