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
    job = _job()  # default status="processing" -- did hold a fal slot
    db = _sweep_db([job])
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    refund = MagicMock()
    release = MagicMock()
    release_slot = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", release)
    monkeypatch.setattr(worker, "release_fal_slot", release_slot)
    # Lock is only released when this was the user's LAST active job (see
    # _fail_stale_job); on a MagicMock db that check would read as truthy.
    monkeypatch.setattr(worker.generation_svc, "_user_has_active_generation_job", lambda db_, uid: False)

    _run(worker.sweep_stale_generation_jobs({}))

    assert job.status == "failed"
    assert "timed out" in job.error_message.lower()
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)  # exact stored split
    release.assert_called_once_with(job.user_id)
    release_slot.assert_called_once_with(job.team_id)  # was "processing" -- had a real slot to free
    db.commit.assert_called_once()


def test_sweep_does_not_release_a_fal_slot_for_a_job_that_never_reserved_one(monkeypatch):
    """Regression test: a job swept while still 'queued' (worker never picked
    it up, or bailed early e.g. on an unknown tool) never reserved a fal slot
    -- try_reserve_fal_slot only runs once a job is dispatched into
    submit_generation_to_fal. Calling release_fal_slot for it anyway would
    decrement a counter nothing incremented, corrupting the concurrency cap
    (reproduced live: the real dev Redis inflight counter was found at -14
    from exactly this kind of unmatched release)."""
    job = _job(status="queued")
    db = _sweep_db([job])
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "refund_credits", MagicMock())
    monkeypatch.setattr(worker, "release_generation_lock", MagicMock())
    release_slot = MagicMock()
    monkeypatch.setattr(worker, "release_fal_slot", release_slot)

    _run(worker.sweep_stale_generation_jobs({}))

    assert job.status == "failed"
    release_slot.assert_not_called()


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
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

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
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.sweep_stale_generation_jobs({}))

    populate_mock.assert_called_once()
    assert job.status == "failed"


# --------------------------------------------------------------------------- #
# END TO END: real sweep + real refund_credits + real balances
#
# The tests above stub refund_credits and only assert it was CALLED with the
# right numbers. Nothing proved the property that actually matters after a
# worker freeze: a job stuck past the timeout ends up status=failed AND the
# team's credit balances are genuinely restored, in the right pools, exactly
# once -- while jobs that aren't stuck are left alone.
#
# Runs against an isolated in-memory SQLite database (never the shared dev
# Postgres: the sweep scans EVERY stale job in whatever DB it is pointed at, so
# running it there would silently fail-and-refund someone's real queued jobs).
# Only the two Redis-backed release_* helpers are stubbed.
# --------------------------------------------------------------------------- #

def _sqlite_sweep_env(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.core.database import Base
    from app.models.generation_job import GenerationJob
    from app.models.team import Team

    @compiles(JSONB, "sqlite")
    def _jsonb_as_json(element, compiler, **kw):
        return "JSON"

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine, tables=[Team.__table__, GenerationJob.__table__])
    Session = sessionmaker(bind=engine)

    monkeypatch.setattr(worker, "SessionLocal", Session)
    released_locks, released_slots = [], []
    monkeypatch.setattr(worker, "release_generation_lock", released_locks.append)
    monkeypatch.setattr(worker, "release_fal_slot", released_slots.append)
    return Session, released_locks, released_slots


def test_sweep_end_to_end_fails_stuck_jobs_and_restores_exact_balances(monkeypatch):
    from app.models.generation_job import GenerationJob
    from app.models.team import Team
    from app.services.generation import SWEEP_TIMEOUT_MESSAGE

    Session, released_locks, released_slots = _sqlite_sweep_env(monkeypatch)
    now = datetime.now(timezone.utc)
    db = Session()

    user_a, user_b, user_c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    # Balance AFTER all five spends below. Started at sub=10 / topup=20:
    #   stuck queued     spent 3 sub + 2 topup
    #   stuck processing spent 0 sub + 4 topup
    #   young queued     spent 0 sub + 3 topup   (9 min -- NOT past the cutoff)
    #   completed        spent 2 sub + 0 topup   (long done -- must never be touched)
    #   stuck queued #2  spent 1 sub + 1 topup   (a second stuck job for the same team)
    team = Team(name="sweep e2e", subscription_credits_remaining=10 - 3 - 2 - 1, topup_credits_balance=20 - 2 - 4 - 3 - 1)
    db.add(team)
    db.flush()

    def job(user, status, age_minutes, sub, topup):
        j = GenerationJob(
            team_id=team.id, user_id=user, feature_type="recolor", status=status,
            input_params={}, credits_charged=sub + topup,
            credits_from_subscription=sub, credits_from_topup=topup,
            created_at=now - timedelta(minutes=age_minutes),
        )
        db.add(j)
        db.flush()
        return j.id

    stuck_queued = job(user_a, "queued", 11, 3, 2)
    stuck_processing = job(user_b, "processing", 12, 0, 4)
    young_queued = job(user_c, "queued", 9, 0, 3)
    completed = job(user_a, "completed", 30, 2, 0)
    stuck_queued_2 = job(user_c, "queued", 45, 1, 1)
    db.commit()
    assert (team.subscription_credits_remaining, team.topup_credits_balance) == (4, 10)

    _run(worker.sweep_stale_generation_jobs({}))

    db.expire_all()
    status = {j.id: j for j in db.query(GenerationJob).all()}

    # 1. the three stuck jobs are failed, with the sweep's message and a timestamp
    for stuck in (stuck_queued, stuck_processing, stuck_queued_2):
        assert status[stuck].status == "failed"
        assert status[stuck].error_message == SWEEP_TIMEOUT_MESSAGE
        assert status[stuck].completed_at is not None

    # 2. everything that is NOT stuck is left exactly as it was
    assert status[young_queued].status == "queued"          # 9 min < 10 min cutoff
    assert status[young_queued].error_message is None
    assert status[completed].status == "completed"           # finished jobs never re-failed

    # 3. THE POINT: credits restored to the exact pools they were taken from.
    #    sub: 4 + 3 (stuck_queued) + 0 + 1 (stuck_queued_2) = 8
    #    topup: 10 + 2 + 4 + 1 = 17   (young_queued's 3 and completed's 2 stay spent)
    db.refresh(team)
    assert team.subscription_credits_remaining == 8
    assert team.topup_credits_balance == 17
    # ...which is the original 10/20 minus ONLY the two jobs that legitimately spent:
    assert team.subscription_credits_remaining == 10 - 2      # completed job's 2 sub
    assert team.topup_credits_balance == 20 - 3               # young job's 3 topup

    # 4. per-user generation locks freed for the owners who have NOTHING else
    #    active. user_c is deliberately NOT freed: their stuck job failed, but
    #    they still have `young_queued` genuinely in flight, and that job's
    #    lock must survive (see test_sweep_never_frees_the_lock_of_a_newer_generation).
    assert sorted(map(str, released_locks)) == sorted(map(str, [user_a, user_b]))
    # ...and a fal slot released ONLY for the one job that had reserved one
    #    (processing). Releasing for a never-submitted queued job would corrupt
    #    the concurrency counter.
    assert released_slots == [team.id]

    # 5. a second sweep is a no-op: no double refund, no repeat releases
    _run(worker.sweep_stale_generation_jobs({}))
    db.expire_all()
    db.refresh(team)
    assert (team.subscription_credits_remaining, team.topup_credits_balance) == (8, 17)
    assert len(released_locks) == 2 and len(released_slots) == 1
    db.close()


def test_sweep_end_to_end_a_job_that_finishes_mid_sweep_is_not_refunded(monkeypatch):
    """The other half of 'exactly once': a job a webhook completed between the
    sweep's scan and its lock must NOT be failed or refunded. Uses the real
    locked re-check (_fail_stale_job) against real rows."""
    from app.models.generation_job import GenerationJob
    from app.models.team import Team

    Session, released_locks, released_slots = _sqlite_sweep_env(monkeypatch)
    db = Session()
    team = Team(name="race", subscription_credits_remaining=0, topup_credits_balance=5)
    db.add(team)
    db.flush()
    j = GenerationJob(
        team_id=team.id, user_id=uuid.uuid4(), feature_type="recolor", status="completed",
        input_params={}, credits_charged=2, credits_from_subscription=0, credits_from_topup=2,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=30),
    )
    db.add(j)
    db.commit()

    # the sweep saw it as stale a moment ago; by the time it locks, it's completed
    assert worker._fail_stale_job(db, j.id, "Generation timed out") is False

    db.refresh(team)
    db.refresh(j)
    assert j.status == "completed"
    assert team.topup_credits_balance == 5   # NOT 7 -- no refund for a finished job
    assert released_locks == [] and released_slots == []
    db.close()


def test_sweep_never_frees_the_lock_of_a_newer_generation(monkeypatch):
    """The generation lock is ONE key per user, and it expires on its own (tool
    timeout + 30s) long before the 10-minute sweep runs. So by the time the
    sweep fails an old stuck job, the user may already have started a NEW
    generation under a fresh lock. Releasing unconditionally would free THAT
    lock, silently letting them start a second concurrent generation. The lock
    must only be released once the user has no other active job."""
    from app.models.generation_job import GenerationJob
    from app.models.team import Team

    Session, released_locks, released_slots = _sqlite_sweep_env(monkeypatch)
    now = datetime.now(timezone.utc)
    db = Session()
    team = Team(name="lock scope", subscription_credits_remaining=0, topup_credits_balance=0)
    db.add(team)
    db.flush()
    user = uuid.uuid4()

    def job(status, age_minutes):
        j = GenerationJob(
            team_id=team.id, user_id=user, feature_type="recolor", status=status,
            input_params={}, credits_charged=1, credits_from_subscription=0, credits_from_topup=1,
            created_at=now - timedelta(minutes=age_minutes),
        )
        db.add(j)
        db.flush()
        return j.id

    old_stuck = job("queued", 45)      # the job being swept
    newer_active = job("processing", 1)  # started after the old lock expired
    db.commit()

    _run(worker.sweep_stale_generation_jobs({}))

    db.expire_all()
    assert db.get(GenerationJob, old_stuck).status == "failed"
    assert db.get(GenerationJob, newer_active).status == "processing"   # untouched
    assert released_locks == []   # the newer generation's lock survives

    # once that newer job is the only thing left and it too gets swept/finishes,
    # the lock IS released -- the one-at-a-time rule ends when the user is idle
    db.get(GenerationJob, newer_active).created_at = now - timedelta(minutes=30)
    db.commit()
    _run(worker.sweep_stale_generation_jobs({}))
    assert [str(u) for u in released_locks] == [str(user)]
    db.close()


def test_sweep_frees_the_lock_when_the_stuck_job_was_the_only_one(monkeypatch):
    """The user-visible half: someone who gave up on a stuck job and has
    nothing else running must not stay blocked by 'already in progress'."""
    from app.models.generation_job import GenerationJob
    from app.models.team import Team

    Session, released_locks, released_slots = _sqlite_sweep_env(monkeypatch)
    db = Session()
    team = Team(name="lone stuck", subscription_credits_remaining=0, topup_credits_balance=0)
    db.add(team)
    db.flush()
    user = uuid.uuid4()
    db.add(GenerationJob(
        team_id=team.id, user_id=user, feature_type="recolor", status="queued",
        input_params={}, credits_charged=2, credits_from_subscription=0, credits_from_topup=2,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=15),
    ))
    db.commit()

    _run(worker.sweep_stale_generation_jobs({}))

    assert [str(u) for u in released_locks] == [str(user)]
    db.close()
