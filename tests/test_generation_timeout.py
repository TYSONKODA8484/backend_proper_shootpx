"""check_generation_timeouts: the frequent, per-tool timeout check that runs
far more often than the 10-minute catch-all sweep, timing each "processing"
job out against its OWN tool's generation_timeout_seconds instead of one
fixed cutoff for every tool.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from app import worker


def _run(coro):
    return asyncio.run(coro)


def _job(feature_type="recolor", status="processing", seconds_old=90,
         credits_charged=5, from_sub=3, from_topup=2, fal_request_id=None):
    return MagicMock(
        id=uuid.uuid4(), status=status, feature_type=feature_type,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=seconds_old),
        team_id=uuid.uuid4(), user_id=uuid.uuid4(),
        credits_charged=credits_charged, credits_from_subscription=from_sub,
        credits_from_topup=from_topup, error_message=None, completed_at=None,
        fal_request_id=fal_request_id,
    )


def _tool(timeout_seconds=60, fal_model_id="fake-model"):
    return MagicMock(generation_timeout_seconds=timeout_seconds, fal_model_id=fal_model_id)


def _timeout_db(processing_jobs, tools_by_feature_type, locked_by_id=None):
    """processing_jobs: what the outer status=='processing' scan returns.
    tools_by_feature_type: {feature_type: tool_or_None} for the per-job
    ToolDefinition lookup. locked_by_id: {job_id: job_or_None} for the
    per-job locked re-check inside _fail_stale_job -- defaults to the same
    objects (no concurrent change)."""
    locked_by_id = locked_by_id if locked_by_id is not None else {j.id: j for j in processing_jobs}
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "GenerationJob":
            def filt(*args):
                inner = MagicMock()
                if args[0].left.key == "id":
                    target_id = args[0].right.value
                    inner.populate_existing.return_value.with_for_update.return_value.first.return_value = (
                        locked_by_id.get(target_id)
                    )
                else:
                    inner.all.return_value = processing_jobs
                return inner
            q.filter.side_effect = filt
        elif name == "ToolDefinition":
            def filt(*args):
                target_ft = args[0].right.value
                inner = MagicMock()
                inner.first.return_value = tools_by_feature_type.get(target_ft)
                return inner
            q.filter.side_effect = filt
        return q

    db.query.side_effect = query
    return db


def test_job_exceeding_its_own_tools_timeout_is_failed_and_refunded(monkeypatch):
    job = _job(feature_type="recolor", seconds_old=90)  # recolor's budget is 60s
    db = _timeout_db([job], {"recolor": _tool(60)})
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    refund = MagicMock()
    release = MagicMock()
    release_slot = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", release)
    monkeypatch.setattr(worker, "release_fal_slot", release_slot)
    # The lock is only released when this was the user's LAST active job; on a
    # MagicMock db that check would read as a truthy "yes, still active".
    monkeypatch.setattr(worker.generation_svc, "_user_has_active_generation_job", lambda db_, uid: False)

    _run(worker.check_generation_timeouts({}))

    assert job.status == "failed"
    assert job.error_message == worker.TIMEOUT_MESSAGE
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)  # exact stored split
    release.assert_called_once_with(job.user_id)
    release_slot.assert_called_once_with(job.team_id)  # "processing" -- had a real slot to free


def test_different_tool_with_longer_timeout_is_not_touched_at_same_elapsed_time(monkeypatch):
    """Proves the check is genuinely per-tool, not one global cutoff: a job
    for a tool with a much longer budget must survive the exact elapsed time
    that already fails recolor's job."""
    fast_job = _job(feature_type="recolor", seconds_old=90)      # over recolor's 60s
    slow_job = _job(feature_type="test_tool", seconds_old=90)    # well under test_tool's 300s
    db = _timeout_db(
        [fast_job, slow_job],
        {"recolor": _tool(60), "test_tool": _tool(300)},
    )
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "refund_credits", MagicMock())
    monkeypatch.setattr(worker, "release_generation_lock", MagicMock())
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.check_generation_timeouts({}))

    assert fast_job.status == "failed"
    assert slow_job.status == "processing"  # untouched


def test_job_completing_within_its_timeout_is_untouched(monkeypatch):
    job = _job(feature_type="recolor", seconds_old=10)  # well under recolor's 60s
    db = _timeout_db([job], {"recolor": _tool(60)})
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    refund = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)

    _run(worker.check_generation_timeouts({}))

    assert job.status == "processing"
    refund.assert_not_called()


def test_unknown_tool_is_skipped_not_crashed(monkeypatch):
    """No matching ToolDefinition row (e.g. deleted/renamed mid-flight) must
    not crash the whole check -- the 10-minute sweep will still eventually
    catch a job like this via its own flat cutoff."""
    job = _job(feature_type="ghost_tool", seconds_old=999)
    db = _timeout_db([job], {})
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    refund = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)

    _run(worker.check_generation_timeouts({}))  # must not raise

    assert job.status == "processing"
    refund.assert_not_called()


def test_queued_jobs_are_not_considered_only_processing(monkeypatch):
    """The per-tool check only ever looks at "processing" jobs -- a "queued"
    job hasn't been submitted to fal yet, so there's no tool-specific budget
    to measure it against; the 10-minute sweep's flat cutoff still covers it."""
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

    _run(worker.check_generation_timeouts({}))

    assert captured["args"][0].left.key == "status"
    assert captured["args"][0].right.value == "processing"


def test_one_job_failure_does_not_abort_the_batch(monkeypatch):
    bad = _job(feature_type="recolor", seconds_old=90)
    good = _job(feature_type="recolor", seconds_old=90)
    db = _timeout_db([bad, good], {"recolor": _tool(60)})
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    def refund(db_, team_id, *a, **k):
        if team_id == bad.team_id:
            raise RuntimeError("db blip")
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", MagicMock())
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.check_generation_timeouts({}))

    db.rollback.assert_called()
    assert good.status == "failed"
    db.close.assert_called_once()


# --------------------------------------------------------------------------- #
# Money-loss fix: confirm against fal's real queue status before failing --
# our local timeout elapsing doesn't mean fal's own work did too.
# --------------------------------------------------------------------------- #

def test_fal_confirms_already_completed_resolves_via_real_result_instead_of_failing(monkeypatch):
    """fal reports COMPLETED at the exact moment we'd otherwise time the job
    out -- must fetch the real result and resolve it via the same path a real
    webhook delivery uses (handle_fal_webhook), not fail+refund a job that
    already succeeded."""
    job = _job(feature_type="recolor", seconds_old=90, fal_request_id="fal-req-123")
    tool = _tool(timeout_seconds=60, fal_model_id="fake-model")
    db = _timeout_db([job], {"recolor": tool})
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    check_status = MagicMock(return_value={"status": "COMPLETED"})
    result_payload = {"status": "OK", "payload": {"images": [{"url": "https://fal.test/out.png"}]}}
    fetch_result = MagicMock(return_value=result_payload)
    monkeypatch.setattr(worker, "check_fal_status", check_status)
    monkeypatch.setattr(worker, "fetch_fal_result", fetch_result)

    handle_webhook = MagicMock()
    monkeypatch.setattr(worker.generation_svc, "handle_fal_webhook", handle_webhook)

    refund = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)

    _run(worker.check_generation_timeouts({}))

    check_status.assert_called_once_with("fake-model", "fal-req-123")
    fetch_result.assert_called_once_with("fake-model", "fal-req-123")
    handle_webhook.assert_called_once_with(db, job.id, result_payload)
    refund.assert_not_called()          # the timeout-failure path never ran
    assert job.status == "processing"   # _fail_stale_job was never reached -- handle_fal_webhook (mocked here, tested on its own elsewhere) owns the transition


def test_completed_job_is_delivered_immediately_even_while_still_within_its_budget(monkeypatch):
    """The real local-dev fix: fal's cloud can't reach a PUBLIC_BACKEND_URL of
    127.0.0.1, so the webhook NEVER arrives and this poll is the only thing
    that finishes a job. The COMPLETED check used to sit behind the budget
    gate, so a job fal finished in 40s sat "processing" until its whole budget
    elapsed -- 60s for recolor (looked fine), 180s for creative_photoshoot
    (looked broken). Must deliver as soon as fal says COMPLETED."""
    job = _job(feature_type="creative_photoshoot", seconds_old=20, fal_request_id="fal-req-789")
    tool = _tool(timeout_seconds=180, fal_model_id="fake-model")  # 160s of budget still left
    db = _timeout_db([job], {"creative_photoshoot": tool})
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    monkeypatch.setattr(worker, "check_fal_status", MagicMock(return_value={"status": "COMPLETED"}))
    result_payload = {"status": "OK", "payload": {"images": [{"url": "https://fal.test/out.png"}]}}
    fetch_result = MagicMock(return_value=result_payload)
    monkeypatch.setattr(worker, "fetch_fal_result", fetch_result)
    handle_webhook = MagicMock()
    monkeypatch.setattr(worker.generation_svc, "handle_fal_webhook", handle_webhook)
    refund = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)

    _run(worker.check_generation_timeouts({}))

    fetch_result.assert_called_once_with("fake-model", "fal-req-789")
    handle_webhook.assert_called_once_with(db, job.id, result_payload)
    refund.assert_not_called()  # nothing timed out -- it succeeded well inside its budget


def test_in_progress_job_within_its_budget_is_left_alone_not_failed(monkeypatch):
    """Polling every tick must not become a way to kill jobs early -- fal says
    still working and the budget hasn't elapsed, so nothing happens."""
    job = _job(feature_type="creative_photoshoot", seconds_old=20, fal_request_id="fal-req-abc")
    tool = _tool(timeout_seconds=180, fal_model_id="fake-model")
    db = _timeout_db([job], {"creative_photoshoot": tool})
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    monkeypatch.setattr(worker, "check_fal_status", MagicMock(return_value={"status": "IN_PROGRESS"}))
    fetch_result = MagicMock()
    monkeypatch.setattr(worker, "fetch_fal_result", fetch_result)
    refund = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)

    _run(worker.check_generation_timeouts({}))

    assert job.status == "processing"  # untouched
    fetch_result.assert_not_called()
    refund.assert_not_called()


def test_fal_confirms_still_in_progress_fails_per_the_timeout_and_logs_it(monkeypatch, caplog):
    """fal confirms the request is genuinely still IN_PROGRESS -- the timeout
    failure proceeds exactly as designed, but this specific case must be
    logged distinctly so how often it happens can be measured."""
    job = _job(feature_type="recolor", seconds_old=90, fal_request_id="fal-req-456")
    tool = _tool(timeout_seconds=60, fal_model_id="fake-model")
    db = _timeout_db([job], {"recolor": tool})
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    monkeypatch.setattr(worker, "check_fal_status", MagicMock(return_value={"status": "IN_PROGRESS"}))
    fetch_result = MagicMock()
    monkeypatch.setattr(worker, "fetch_fal_result", fetch_result)
    refund = MagicMock()
    release = MagicMock()
    release_slot = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", release)
    monkeypatch.setattr(worker, "release_fal_slot", release_slot)

    with caplog.at_level("WARNING"):
        _run(worker.check_generation_timeouts({}))

    fetch_result.assert_not_called()   # never fetched a result -- it wasn't COMPLETED
    assert job.status == "failed"      # still fails per the timeout, as designed
    assert job.error_message == worker.TIMEOUT_MESSAGE
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)
    release_slot.assert_called_once_with(job.team_id)
    assert any(
        "still in-progress" in r.message and "IN_PROGRESS" in r.message
        for r in caplog.records
    ), "expected a distinct log line for the still-in-progress case, to measure how often it happens"


def test_unreachable_fal_status_check_falls_back_to_the_local_timeout(monkeypatch):
    """A transient error calling fal's status endpoint must not block the
    safety net -- fall back to the existing local-timeout behavior rather
    than leaving the job stuck."""
    job = _job(feature_type="recolor", seconds_old=90, fal_request_id="fal-req-789")
    tool = _tool(timeout_seconds=60, fal_model_id="fake-model")
    db = _timeout_db([job], {"recolor": tool})
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    def boom(model_id, request_id):
        raise ConnectionError("fal unreachable")
    monkeypatch.setattr(worker, "check_fal_status", boom)
    fetch_result = MagicMock()
    monkeypatch.setattr(worker, "fetch_fal_result", fetch_result)
    refund = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)

    _run(worker.check_generation_timeouts({}))  # must not raise

    fetch_result.assert_not_called()
    assert job.status == "failed"
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)


def test_check_generation_timeouts_is_registered_on_the_worker(monkeypatch):
    """Confirms it's actually wired into the arq worker, not just defined."""
    assert worker.check_generation_timeouts in worker.WorkerSettings.functions
    assert any(
        getattr(job, "coroutine", None) is worker.check_generation_timeouts
        for job in worker.WorkerSettings.cron_jobs
    )
