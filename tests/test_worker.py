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
    """The active pass now makes TWO kinds of TeamSubscription query per row:
    the batch query at the top of the function (1st call), then a per-row
    re-fetch-by-id under .with_for_update() inside the loop (each subsequent
    call, in the same order the loop iterates `due`). A counter over
    db.query(TeamSubscription) calls tells them apart instead of trying to
    parse the actual SQLAlchemy filter expression."""
    db = MagicMock()
    due = list(due)
    seen = {"team_sub_calls": 0}

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            seen["team_sub_calls"] += 1
            call_index = seen["team_sub_calls"]
            if call_index == 1:
                q.filter.return_value.all.return_value = due                # the batch query
            else:
                row_index = call_index - 2
                row = due[row_index] if 0 <= row_index < len(due) else None
                (q.filter.return_value.populate_existing.return_value
                   .with_for_update.return_value.first.return_value) = row
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
    ts = MagicMock(team_id="team-1", subscription_id="sub-1", status="active",
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
        ts = MagicMock(team_id="t", subscription_id="s", status="active", credits_per_refill=10,
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
    bad = MagicMock(team_id="bad", subscription_id="s", status="active", credits_per_refill=10,
                    next_refill_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    good = MagicMock(team_id="good", subscription_id="s", status="active", credits_per_refill=10,
                     next_refill_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    db = _refill_db([bad, good], MagicMock(period_label="week"))
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.refill_due_subscriptions({}))

    db.rollback.assert_called()                 # the bad one was rolled back
    assert isinstance(good.next_refill_at, datetime)   # the good one still processed
    db.close.assert_called_once()               # session always closed


def test_refill_locked_sub_re_query_uses_populate_existing(monkeypatch):
    """Regression test for the identity-map staleness bug found live during
    the sweep_stale_generation_jobs audit and confirmed to affect this same
    function: `due` (the unlocked batch query) loads TeamSubscription rows
    into this session's identity map, so the per-row locked re-query for the
    same id must use populate_existing() -- otherwise it silently returns the
    same stale cached object instead of the fresh, lock-guaranteed row, and a
    subscription cancelled by a concurrent webhook moments earlier would
    still read as 'active' and get refilled anyway."""
    monkeypatch.setattr(worker, "refill_subscription_credits", lambda *a, **k: None)
    ts = MagicMock(team_id="team-1", subscription_id="sub-1", status="active",
                   credits_per_refill=1000, next_refill_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    db = MagicMock()
    call_count = {"n": 0}
    populate_mock = MagicMock()
    populate_mock.return_value.with_for_update.return_value.first.return_value = ts

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            call_count["n"] += 1
            if call_count["n"] == 1:
                q.filter.return_value.all.return_value = [ts]          # outer batch query
                q.join.return_value.filter.return_value.all.return_value = []  # lapse pass
            else:
                q.filter.return_value.populate_existing = populate_mock  # per-row locked re-query
        elif name == "Subscription":
            q.filter.return_value.first.return_value = MagicMock(period_label="month")
        return q

    db.query.side_effect = query
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.refill_due_subscriptions({}))

    populate_mock.assert_called_once()
    assert ts.next_refill_at > datetime(2020, 1, 1, tzinfo=timezone.utc)


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
# subscription.completed -> scheduler's existing lapse-pass, end to end
# --------------------------------------------------------------------------- #

def test_completed_then_scheduled_lapse_zeros_credits_end_to_end(monkeypatch):
    """subscription.completed (webhooks.py) only ever flips status to
    'cancelled' — it never touches next_refill_at. This confirms that's
    sufficient: the SAME next_refill_at the row already carried from
    activation/its last refill is exactly what the scheduler's existing
    lapse-pass uses to decide when to zero the leftover pool, so the two
    compose correctly with zero additional code."""
    monkeypatch.setattr(worker, "refill_subscription_credits", lambda *a, **k: None)

    stale = datetime(2020, 1, 1, tzinfo=timezone.utc)   # already past its natural reset
    team_sub = MagicMock(
        team_id="team-x", status="active",
        razorpay_subscription_id="sub_final", next_refill_at=stale,
    )
    completed_db = MagicMock()
    (completed_db.query.return_value.filter.return_value
        .with_for_update.return_value.first.return_value) = team_sub

    webhooks_svc.handle_subscription_completed(
        completed_db,
        {"event": "subscription.completed",
         "payload": {"subscription": {"entity": {"id": "sub_final"}}}},
    )
    assert team_sub.status == "cancelled"
    completed_db.commit.assert_called_once()

    # the daily scheduler now runs and finds this same row past its reset date
    team = MagicMock(subscription_credits_remaining=42, topup_credits_balance=10)
    db = _refill_db(lapse_due=[team_sub], team_for_update=team)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.refill_due_subscriptions({}))

    assert team.subscription_credits_remaining == 0
    assert team.topup_credits_balance == 10          # untouched


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
# send_yearly_renewal_notices -- yearly plans have no second "charged"
# webhook to key a notice off (total_count=1), so this checks
# current_period_end directly instead of paid_count. Real fix for: yearly
# subscribers previously got NO advance warning before losing access, while
# week/month subscribers did (see handle_subscription_charged's own
# paid_count-based notice in app/services/webhooks.py).
# --------------------------------------------------------------------------- #

def _yearly_notice_db(due=()):
    """Same call-counting trick as _refill_db: the batch query (a .join(...)
    off TeamSubscription) is the 1st db.query(TeamSubscription) call, then
    each subsequent call is the per-row re-fetch-by-id under
    .with_for_update(), in loop order."""
    db = MagicMock()
    due = list(due)
    seen = {"team_sub_calls": 0}

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            seen["team_sub_calls"] += 1
            call_index = seen["team_sub_calls"]
            if call_index == 1:
                q.join.return_value.filter.return_value.all.return_value = due
            else:
                row_index = call_index - 2
                row = due[row_index] if 0 <= row_index < len(due) else None
                (q.filter.return_value.populate_existing.return_value
                   .with_for_update.return_value.first.return_value) = row
        return q

    db.query.side_effect = query
    return db


def test_yearly_renewal_notice_fires_exactly_30_days_out(monkeypatch):
    sent = []
    monkeypatch.setattr(worker, "send_renewal_notice_email", lambda db, ts: sent.append(ts))
    now = datetime.now(timezone.utc)
    ts = MagicMock(
        team_id="team-1", status="active", renewal_notice_sent_at=None,
        current_period_end=now + timedelta(days=30),
    )
    db = _yearly_notice_db([ts])
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.send_yearly_renewal_notices({}))

    assert sent == [ts]
    assert ts.renewal_notice_sent_at is not None
    db.commit.assert_called_once()
    db.close.assert_called_once()


def test_yearly_renewal_notice_does_not_fire_31_days_out(monkeypatch):
    """31 days out is past the window -- the function's own re-check (not
    just the SQL WHERE, which a mocked db can't exercise) must still skip a
    row that somehow reached the loop outside the window."""
    sent = []
    monkeypatch.setattr(worker, "send_renewal_notice_email", lambda db, ts: sent.append(ts))
    now = datetime.now(timezone.utc)
    ts = MagicMock(
        team_id="team-1", status="active", renewal_notice_sent_at=None,
        current_period_end=now + timedelta(days=31),
    )
    db = _yearly_notice_db([ts])
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.send_yearly_renewal_notices({}))

    assert sent == []
    assert ts.renewal_notice_sent_at is None
    db.commit.assert_not_called()


def test_yearly_renewal_notice_is_not_resent_once_already_sent(monkeypatch):
    sent = []
    monkeypatch.setattr(worker, "send_renewal_notice_email", lambda db, ts: sent.append(ts))
    now = datetime.now(timezone.utc)
    already_sent_at = now - timedelta(days=1)
    ts = MagicMock(
        team_id="team-1", status="active", renewal_notice_sent_at=already_sent_at,
        current_period_end=now + timedelta(days=29),
    )
    db = _yearly_notice_db([ts])
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.send_yearly_renewal_notices({}))

    assert sent == []
    assert ts.renewal_notice_sent_at == already_sent_at  # untouched
    db.commit.assert_not_called()


def test_yearly_renewal_notice_skips_a_non_active_row(monkeypatch):
    sent = []
    monkeypatch.setattr(worker, "send_renewal_notice_email", lambda db, ts: sent.append(ts))
    now = datetime.now(timezone.utc)
    ts = MagicMock(
        team_id="team-1", status="cancelled", renewal_notice_sent_at=None,
        current_period_end=now + timedelta(days=10),
    )
    db = _yearly_notice_db([ts])
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.send_yearly_renewal_notices({}))

    assert sent == []


def test_yearly_renewal_notice_query_filters_on_year_period_and_active_status(monkeypatch):
    """Confirms the batch query actually joins Subscription and filters
    period_label == 'year' (not week/month) and status == 'active' -- via the
    real filter args, same pattern as test_lapse_query_targets_cancelled_past_
    reset_with_credits above."""
    captured = {}
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            def jf(*args):
                captured["args"] = args
                m = MagicMock()
                m.all.return_value = []
                return m
            q.join.return_value.filter.side_effect = jf
        return q

    db.query.side_effect = query
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.send_yearly_renewal_notices({}))

    sql = " ".join(str(a) for a in captured["args"])
    assert "team_subscriptions.status =" in sql
    assert "subscription.period_label =" in sql
    assert "team_subscriptions.current_period_end <=" in sql
    assert "team_subscriptions.renewal_notice_sent_at IS NULL" in sql


def test_yearly_renewal_notice_one_failure_does_not_abort_the_batch(monkeypatch):
    def send(db, ts):
        if ts.team_id == "bad":
            raise ValueError("SMTP exploded")
    monkeypatch.setattr(worker, "send_renewal_notice_email", send)
    now = datetime.now(timezone.utc)
    bad = MagicMock(team_id="bad", status="active", renewal_notice_sent_at=None,
                    current_period_end=now + timedelta(days=5))
    good = MagicMock(team_id="good", status="active", renewal_notice_sent_at=None,
                     current_period_end=now + timedelta(days=5))
    db = _yearly_notice_db([bad, good])
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    _run(worker.send_yearly_renewal_notices({}))

    db.rollback.assert_called_once()                    # the bad one rolled back
    assert good.renewal_notice_sent_at is not None       # the good one still processed
    db.close.assert_called_once()


def test_yearly_renewal_notice_reset_on_a_fresh_subscribe_cycle(monkeypatch):
    """create_subscription_checkout must clear renewal_notice_sent_at on a
    resubscribe -- otherwise a team resubscribing after a yearly plan
    completed would carry over the OLD cycle's sent_at and never get warned
    before the new cycle also ends."""
    from app.services import billing as billing_svc
    from app.models.team_subscription import TeamSubscription

    plan = MagicMock(id="plan-1", razorpay_plan_id="plan_rzp_1", period_label="year")
    existing_row = TeamSubscription(team_id="team-1")
    existing_row.status = "cancelled"
    existing_row.renewal_notice_sent_at = datetime.now(timezone.utc) - timedelta(days=200)

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = plan
    db.query.return_value.filter.return_value.with_for_update.return_value.first.return_value = existing_row
    monkeypatch.setattr(
        billing_svc.razorpay_client.subscription, "create",
        lambda *a, **k: {"id": "sub_new", "short_url": "https://rzp.test/x"},
    )

    billing_svc.create_subscription_checkout(db, "team-1", "plan-1")

    assert existing_row.renewal_notice_sent_at is None


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
    db.query.return_value.filter.return_value.with_for_update.return_value.first.return_value = team_sub
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
