"""sweep_stale_generation_jobs: catches jobs stuck in queued/processing past
the timeout, and in particular the webhook-vs-sweep race condition found live
during this audit -- see test_sweep_populate_existing_prevents_stale_identity_map_read
for the root-cause proof."""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from app import worker


def _run(coro):
    return asyncio.run(coro)


def _sweep_db(stale_jobs, locked_by_id=None):
    """locked_by_id: {job_id: job_or_None} -- what the per-job populate_existing()
    .with_for_update() re-query returns for each id. Defaults to returning the
    same object found in stale_jobs (simulating no concurrent change).

    GenerationJob is queried twice per job in the real code: once (call #1)
    for the outer unlocked batch scan, then once per job (calls #2+) for the
    locked re-check -- distinguished here by call order, since both queries
    target the same model."""
    locked_by_id = locked_by_id or {j.id: j for j in stale_jobs}
    db = MagicMock()
    call_count = {"n": 0}

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "GenerationJob":
            call_count["n"] += 1
            if call_count["n"] == 1:
                q.filter.return_value.all.return_value = stale_jobs
            else:
                def by_id_filter(*args):
                    # extract the id being filtered on from GenerationJob.id == X
                    target_id = args[0].right.value
                    inner = MagicMock()
                    inner.populate_existing.return_value.with_for_update.return_value.first.return_value = (
                        locked_by_id.get(target_id)
                    )
                    return inner
                q.filter.side_effect = by_id_filter
        return q

    db.query.side_effect = query
    return db


def _job(status="processing", minutes_old=20, credits_charged=5, from_sub=3, from_topup=2):
    return MagicMock(
        id=uuid.uuid4(), status=status,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_old),
        team_id=uuid.uuid4(), user_id=uuid.uuid4(),
        credits_charged=credits_charged, credits_from_subscription=from_sub, credits_from_topup=from_topup,
        error_message=None, completed_at=None,
    )


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #

def test_sweep_marks_stale_job_failed_refunds_and_releases_lock(monkeypatch):
    job = _job()
    db = _sweep_db([job])
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", release)

    _run(worker.sweep_stale_generation_jobs({}))

    assert job.status == "failed"
    assert "timed out" in job.error_message.lower()
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)  # exact stored split
    release.assert_called_once_with(job.user_id)
    db.commit.assert_called_once()


def test_sweep_skips_job_already_resolved_by_the_time_it_is_locked(monkeypatch):
    """The re-check itself: if the locked re-query shows a terminal status
    (simulating a webhook having already resolved it), the sweep must not
    touch it."""
    job = _job()
    resolved_job = MagicMock(status="completed")
    db = _sweep_db([job], locked_by_id={job.id: resolved_job})
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", release)

    _run(worker.sweep_stale_generation_jobs({}))

    refund.assert_not_called()
    release.assert_not_called()
    db.commit.assert_not_called()


def test_sweep_skips_job_that_disappeared_before_locking(monkeypatch):
    job = _job()
    db = _sweep_db([job], locked_by_id={job.id: None})
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    refund = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", MagicMock())

    _run(worker.sweep_stale_generation_jobs({}))  # must not raise

    refund.assert_not_called()


def test_sweep_one_job_failure_does_not_abort_the_batch(monkeypatch):
    bad = _job()
    good = _job()
    db = _sweep_db([bad, good])
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    def refund(db_, team_id, *a, **k):
        if team_id == bad.team_id:
            raise RuntimeError("db blip")
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", MagicMock())

    _run(worker.sweep_stale_generation_jobs({}))

    db.rollback.assert_called()          # the bad one rolled back
    assert good.status == "failed"       # the good one still processed
    db.close.assert_called_once()


def test_sweep_query_targets_the_right_statuses_and_10_minute_cutoff(monkeypatch):
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

    before = datetime.now(timezone.utc)
    _run(worker.sweep_stale_generation_jobs({}))
    after = datetime.now(timezone.utc)

    sql = " ".join(str(a) for a in captured["args"])
    assert "generation_jobs.status IN" in sql
    assert "generation_jobs.created_at <=" in sql

    # the cutoff bind value is ~10 minutes before "now" at call time
    cutoff_arg = captured["args"][1]
    cutoff_value = cutoff_arg.right.value
    assert before - timedelta(minutes=10, seconds=2) <= cutoff_value <= after - timedelta(minutes=10) + timedelta(seconds=2)


# --------------------------------------------------------------------------- #
# THE race condition: root-cause proof for the bug found live during this
# audit, and confirmation that populate_existing() is actually wired in.
# --------------------------------------------------------------------------- #

def test_sweep_populate_existing_prevents_stale_identity_map_read(tmp_path):
    """Root-cause reproduction of the real bug found live against the actual
    dev Postgres DB: sweep_stale_generation_jobs loads jobs into its session's
    identity map via an UNLOCKED query, then re-queries the SAME id on the
    SAME session under with_for_update(). Without populate_existing(),
    SQLAlchemy returns the SAME cached Python object with its stale, pre-lock
    attributes -- NOT the fresh row the lock just fetched -- silently
    defeating the "re-check status after locking" safety pattern. Proven live:
    a job a concurrent webhook had already marked 'completed' still read as
    'processing' here, moments after the webhook's commit, causing the sweep
    to incorrectly re-fail and double-refund an already-succeeded job.

    This test reproduces the exact SQLAlchemy mechanism in isolation (no
    threads, no Postgres, no network) with a minimal standalone model -- the
    identity-map behavior is generic to SQLAlchemy, not specific to this
    schema or database backend.
    """
    from sqlalchemy import create_engine, Column, Text
    from sqlalchemy.dialects.postgresql import UUID
    from sqlalchemy.orm import sessionmaker, declarative_base

    TestBase = declarative_base()

    class Row(TestBase):
        __tablename__ = "identity_map_test_rows"
        id = Column(UUID(as_uuid=True), primary_key=True)
        status = Column(Text)

    engine = create_engine(f"sqlite:///{tmp_path/'test.db'}")
    TestBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    session_a = Session()  # stands in for the sweep's own session
    row_id = uuid.uuid4()
    session_a.add(Row(id=row_id, status="processing"))
    session_a.commit()

    # the sweep's outer, UNLOCKED query -- loads this id into session_a's identity map
    loaded = session_a.query(Row).filter(Row.id == row_id).all()
    assert loaded[0].status == "processing"

    # a concurrent transaction (e.g. the real webhook, on its own session)
    # resolves the row while the sweep is still mid-flight
    session_b = Session()
    concurrent_row = session_b.query(Row).filter(Row.id == row_id).first()
    concurrent_row.status = "completed"
    session_b.commit()
    session_b.close()

    # WITHOUT populate_existing(): reproduces the bug -- stale cached data
    stale_reread = session_a.query(Row).filter(Row.id == row_id).first()
    assert stale_reread.status == "processing", (
        "if this ever changes, SQLAlchemy's default identity-map caching "
        "behavior has changed and the reasoning behind the populate_existing() "
        "fix in sweep_stale_generation_jobs should be re-verified"
    )

    # WITH populate_existing(): the actual fix -- reflects the fresh row
    fresh_reread = session_a.query(Row).filter(Row.id == row_id).populate_existing().first()
    assert fresh_reread.status == "completed"

    session_a.close()
    engine.dispose()


def test_sweep_re_lock_query_actually_calls_populate_existing(monkeypatch):
    """Confirms the fix is really wired into sweep_stale_generation_jobs
    itself, not just proven as a general SQLAlchemy fact above."""
    job = _job()
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [job]
    populate_mock = db.query.return_value.filter.return_value.populate_existing
    populate_mock.return_value.with_for_update.return_value.first.return_value = job
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "refund_credits", MagicMock())
    monkeypatch.setattr(worker, "release_generation_lock", MagicMock())

    _run(worker.sweep_stale_generation_jobs({}))

    populate_mock.assert_called_once()
    assert job.status == "failed"
