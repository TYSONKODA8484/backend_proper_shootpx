"""Generation pipeline: per-user lock, credit reserve/refund, webhook idempotency,
and ownership checks on /generate and /jobs/{id}."""

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.cache import redis_client
from app.core.database import get_db
from app.deps import get_current_user
from app.main import app
from app.models.generation_job import GenerationJob
from app.services import generation as generation_svc
from app.services.generation_lock import (
    LOCK_TTL_SECONDS,
    acquire_generation_lock,
    release_generation_lock,
)

TEAM_ID = uuid.uuid4()


# --------------------------------------------------------------------------- #
# Lock behavior — against the real Redis instance, same pattern as
# test_cache_admin.py uses for the real cache module. This is the actual
# mechanism (SET NX EX), not a mock of it.
# --------------------------------------------------------------------------- #

@pytest.fixture
def lock_user():
    user_id = uuid.uuid4()
    yield user_id
    release_generation_lock(user_id)  # always clean up, pass or fail


def test_lock_blocks_second_acquire_for_same_user(lock_user):
    assert acquire_generation_lock(lock_user) is True
    assert acquire_generation_lock(lock_user) is False  # already held


def test_lock_is_independent_per_user(lock_user):
    other_user = uuid.uuid4()
    try:
        assert acquire_generation_lock(lock_user) is True
        assert acquire_generation_lock(other_user) is True  # not blocked by lock_user's lock
    finally:
        release_generation_lock(other_user)


def test_release_then_acquire_succeeds_again(lock_user):
    assert acquire_generation_lock(lock_user) is True
    release_generation_lock(lock_user)
    assert acquire_generation_lock(lock_user) is True


def test_lock_has_a_ttl_safety_net(lock_user):
    """Must have a TTL so a crash mid-flow can never lock a user out forever."""
    acquire_generation_lock(lock_user)
    ttl = redis_client.ttl(f"genlock:user:{lock_user}")
    assert 0 < ttl <= LOCK_TTL_SECONDS


def test_release_is_a_safe_noop_when_nothing_held(lock_user):
    release_generation_lock(lock_user)  # never acquired -- must not raise


# --------------------------------------------------------------------------- #
# create_generation_batch — lock + credit reserve, service level (mocked DB)
# --------------------------------------------------------------------------- #

def _tool(feature_type="test_tool", credit_cost=5, active=True, default_output_count=1):
    return MagicMock(feature_type=feature_type, credit_cost_per_output=credit_cost,
                      is_active=active, default_output_count=default_output_count)


def _job_db(tool):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = tool
    return db


def test_create_batch_rejected_when_lock_already_held(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: False)
    spend = MagicMock()
    monkeypatch.setattr(generation_svc, "spend_credits", spend)
    db = _job_db(_tool())

    with pytest.raises(ValueError, match="already have a generation in progress"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {})

    spend.assert_not_called()  # lock is checked before anything else touches credits


def test_create_batch_releases_lock_on_unknown_tool(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    user_id = uuid.uuid4()
    db = _job_db(None)  # tool lookup returns nothing

    with pytest.raises(ValueError, match="Unknown tool"):
        generation_svc.create_generation_batch(db, TEAM_ID, user_id, "nope", {})

    assert released == [user_id]


def test_create_batch_releases_lock_on_inactive_tool(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    user_id = uuid.uuid4()
    db = _job_db(_tool(active=False))

    with pytest.raises(ValueError, match="not currently available"):
        generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {})

    assert released == [user_id]


def test_create_batch_releases_lock_on_insufficient_credits(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(
        generation_svc, "spend_credits",
        MagicMock(side_effect=ValueError("Insufficient credits. Available: 0, needed: 5")),
    )
    user_id = uuid.uuid4()
    db = _job_db(_tool())

    with pytest.raises(ValueError, match="Insufficient credits"):
        generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {})

    assert released == [user_id]
    db.rollback.assert_called_once()


def test_create_batch_success_charges_and_keeps_lock_held(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(
        generation_svc, "spend_credits",
        MagicMock(return_value=(MagicMock(), 3, 2)),  # split across both pools, real ints
    )
    user_id = uuid.uuid4()
    db = _job_db(_tool(credit_cost=5))

    jobs = generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {"prompt": "x"}, 1)

    assert len(jobs) == 1
    job = jobs[0]
    assert job.status == "queued"
    assert job.credits_charged == 5
    assert job.credits_from_subscription == 3
    assert job.credits_from_topup == 2
    db.add.assert_called_once_with(job)
    db.commit.assert_called_once()  # spend + job insert land in ONE transaction
    assert released == []  # lock stays held until the webhook resolves the job


def test_create_batch_spends_credits_without_committing_separately(monkeypatch):
    """Regression test: spend_credits must be called with commit=False so a
    failure creating the job rows can never leave credits spent with no job to
    account for them (see the credit-loss bug this closes)."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    spend = MagicMock(return_value=(MagicMock(), 5, 0))  # real ints, needed for the // split math
    monkeypatch.setattr(generation_svc, "spend_credits", spend)
    db = _job_db(_tool(credit_cost=5))

    generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {})

    assert spend.call_args.kwargs.get("commit") is False


def test_create_batch_rolls_back_and_releases_lock_if_job_insert_fails(monkeypatch):
    """The actual credit-loss bug found during an earlier audit: spend_credits
    used to commit on its own, so a failure building/inserting a job row
    afterward left credits permanently spent with no job row and no way to
    refund them. Reproduced live against the real dev DB, then fixed by
    making the spend and the job insert(s) one transaction (commit=False +
    a single db.commit())."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "spend_credits", MagicMock(return_value=(MagicMock(), 5, 0)))
    user_id = uuid.uuid4()
    db = _job_db(_tool(credit_cost=5))
    db.commit.side_effect = RuntimeError("db blip during job insert")

    with pytest.raises(RuntimeError):
        generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {})

    db.rollback.assert_called_once()  # the spend is rolled back along with the failed insert(s)
    assert released == [user_id]      # user is never left stuck


# --------------------------------------------------------------------------- #
# create_generation_batch — the batching logic itself (multi-output requests)
# --------------------------------------------------------------------------- #

def test_create_batch_of_five_creates_five_jobs_sharing_one_batch_id(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits", MagicMock(return_value=(MagicMock(), 25, 0)))
    db = _job_db(_tool(credit_cost=5))

    jobs = generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {}, 5)

    assert len(jobs) == 5
    batch_ids = {j.batch_id for j in jobs}
    assert len(batch_ids) == 1  # all 5 share exactly one batch_id
    assert db.add.call_count == 5
    db.commit.assert_called_once()  # one atomic transaction for the whole batch


def test_create_batch_splits_total_credit_charge_across_all_jobs_with_remainder_on_first(monkeypatch):
    """total_cost = 5 credits/output * 5 outputs = 25. Split unevenly on
    purpose (23 from subscription, 7 from topup -- neither divides evenly by
    5) so the remainder-handling is actually exercised, not just the clean
    case."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits", MagicMock(return_value=(MagicMock(), 23, 7)))
    db = _job_db(_tool(credit_cost=5))

    jobs = generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {}, 5)

    assert len(jobs) == 5
    # the sum across all 5 jobs must equal exactly what was charged, split or not
    assert sum(j.credits_from_subscription for j in jobs) == 23
    assert sum(j.credits_from_topup for j in jobs) == 7
    assert sum(j.credits_charged for j in jobs) == 25
    # even split is 23//5=4 remainder 3, and 7//5=1 remainder 2 -- both
    # remainders land on the first job only
    assert jobs[0].credits_from_subscription == 4 + 3
    assert jobs[0].credits_from_topup == 1 + 2
    for j in jobs[1:]:
        assert j.credits_from_subscription == 4
        assert j.credits_from_topup == 1
    # each job's own credits_charged is still the flat per-output cost, not a split
    assert all(j.credits_charged == 5 for j in jobs)


def test_create_batch_uses_tool_default_output_count_when_not_specified(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    spend = MagicMock(return_value=(MagicMock(), 15, 0))
    monkeypatch.setattr(generation_svc, "spend_credits", spend)
    db = _job_db(_tool(credit_cost=5, default_output_count=3))

    jobs = generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {}, output_count=0)

    assert len(jobs) == 3
    spend.assert_called_once_with(db, TEAM_ID, 15, commit=False)  # 5 * 3, not 5 * 0


# --------------------------------------------------------------------------- #
# handle_fal_webhook — success/failure/refund-split/idempotent replay
# --------------------------------------------------------------------------- #

def _job(status="queued", credits_charged=5, from_sub=3, from_topup=2, batch_id=None):
    return MagicMock(
        status=status,
        credits_charged=credits_charged,
        credits_from_subscription=from_sub,
        credits_from_topup=from_topup,
        team_id=TEAM_ID,
        user_id=uuid.uuid4(),
        batch_id=batch_id or uuid.uuid4(),
        output_url=None,
        error_message=None,
    )


def _webhook_db(job, remaining_in_batch=0):
    """remaining_in_batch: how many OTHER jobs in the same batch are still
    queued/processing -- 0 or 1 means "this was the last one" (the count
    includes this job's own not-yet-flushed row, see fail_and_release /
    handle_fal_webhook's `remaining <= 1` check), anything higher means
    siblings are still in flight and the lock must stay held."""
    db = MagicMock()
    (db.query.return_value.filter.return_value
       .with_for_update.return_value.first.return_value) = job
    db.query.return_value.filter.return_value.count.return_value = remaining_in_batch
    return db


def test_webhook_success_completes_job_and_releases_lock(monkeypatch):
    """Storage integration (see tests/test_storage_integration.py for the full
    download/upload matrix): output_url ends up as OUR permanent storage URL,
    not fal's raw temporary one."""
    job = _job()
    db = _webhook_db(job)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage",
                         MagicMock(return_value="https://our-storage.test/permanent/out.png"))
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"images": [{"url": "https://x.test/out.png"}]}},
    )

    assert job.status == "completed"
    assert job.output_url == "https://our-storage.test/permanent/out.png"
    assert released == [job.user_id]
    refund.assert_not_called()
    db.commit.assert_called_once()


def test_webhook_failure_sets_failed_and_refunds_matching_split(monkeypatch):
    job = _job(from_sub=3, from_topup=2, credits_charged=5)
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "ERROR", "error": "boom"})

    assert job.status == "failed"
    assert job.error_message == "boom"
    refund.assert_called_once_with(db, TEAM_ID, 5, 3, 2)  # exact pools, exact split
    db.commit.assert_called_once()


def test_webhook_failure_defaults_error_message_when_absent(monkeypatch):
    job = _job()
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "refund_credits", MagicMock())

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "ERROR"})

    assert job.status == "failed"
    assert job.error_message == "Generation failed"


def test_webhook_unknown_job_id_is_a_safe_noop(monkeypatch):
    db = MagicMock()
    (db.query.return_value.filter.return_value
       .with_for_update.return_value.first.return_value) = None
    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "OK", "payload": {}})

    refund.assert_not_called()
    release.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.parametrize("terminal_status", ["completed", "failed"])
def test_webhook_replay_on_terminal_job_is_a_safe_noop(monkeypatch, terminal_status):
    """Idempotent replay: a webhook delivered twice (or replayed by fal) for a
    job that's already terminal must not re-set fields, re-refund, or
    re-release an already-released lock in a way that breaks anything."""
    job = _job(status=terminal_status, from_sub=3, from_topup=2)
    job.output_url = "https://original.test/out.png"
    job.error_message = "original failure" if terminal_status == "failed" else None
    db = _webhook_db(job)
    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"images": [{"url": "https://REPLAY-SHOULD-NOT-APPLY.test/x.png"}]}},
    )

    # nothing changed, nothing re-triggered
    assert job.output_url == "https://original.test/out.png"
    assert job.error_message == ("original failure" if terminal_status == "failed" else None)
    refund.assert_not_called()
    release.assert_not_called()
    db.commit.assert_not_called()


def test_webhook_no_images_in_success_payload_fails_and_refunds(monkeypatch):
    """Updated for the storage-integration behavior change: fal reporting
    success with no image is not a usable result, so it now fails + refunds
    (see tests/test_storage_integration.py for the full matrix) instead of
    completing with a null output_url."""
    job = _job()
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "OK", "payload": {"images": []}})

    assert job.status == "failed"
    assert job.output_url is None
    refund.assert_called_once()


# --------------------------------------------------------------------------- #
# Batch-aware lock release: held until every job in the batch is terminal
# --------------------------------------------------------------------------- #

def test_lock_stays_held_after_only_first_of_five_jobs_completes(monkeypatch):
    job = _job(status="queued")
    db = _webhook_db(job, remaining_in_batch=4)  # 4 siblings still queued/processing
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage",
                         MagicMock(return_value="https://our-storage.test/1.png"))

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"images": [{"url": "https://fal.test/1.png"}]}},
    )

    assert job.status == "completed"
    release.assert_not_called()  # siblings still in flight -- lock must stay held


def test_lock_releases_once_the_last_job_in_the_batch_completes(monkeypatch):
    job = _job(status="queued")
    # 1, not 0: the count includes this job's own not-yet-flushed row (autoflush
    # is off), so "1 remaining" IS "this was the last one" -- see handle_fal_webhook.
    db = _webhook_db(job, remaining_in_batch=1)
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage",
                         MagicMock(return_value="https://our-storage.test/5.png"))

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"images": [{"url": "https://fal.test/5.png"}]}},
    )

    assert job.status == "completed"
    release.assert_called_once_with(job.user_id)


def test_lock_release_check_counts_only_queued_and_processing_siblings(monkeypatch):
    """A failure mid-batch must be able to release the lock too, once it was
    the last non-terminal job -- not just successes."""
    job = _job(status="processing")
    db = _webhook_db(job, remaining_in_batch=1)
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)
    monkeypatch.setattr(generation_svc, "refund_credits", MagicMock())

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "ERROR", "error": "boom"})

    assert job.status == "failed"
    release.assert_called_once_with(job.user_id)


# --------------------------------------------------------------------------- #
# Route-level ownership checks: /generate and /jobs/{id}
# --------------------------------------------------------------------------- #

@pytest.fixture
def gen_client(monkeypatch):
    fake_user = MagicMock(id=uuid.uuid4())
    db = MagicMock()

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db

    yield TestClient(app, raise_server_exceptions=False), fake_user, db
    app.dependency_overrides.clear()


def test_generate_rejects_malformed_team_id_with_422_not_500(gen_client):
    """Regression test: team_id used to be typed `str` on the request model, so
    a malformed value sailed past validation and hit the DB as a raw UUID
    comparison, raising an unhandled 500. Every other route in this codebase
    types team_id as UUID; this one now matches."""
    client, fake_user, db = gen_client

    res = client.post(
        "/generate",
        json={"team_id": "not-a-uuid", "feature_type": "test_tool", "input_params": {}},
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 422


def test_generate_rejects_non_team_member(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: False)

    res = client.post(
        "/generate",
        json={"team_id": str(TEAM_ID), "feature_type": "test_tool", "input_params": {}},
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 403


def test_generate_allows_team_member_and_surfaces_value_error_as_400(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        MagicMock(side_effect=ValueError("You already have a generation in progress. Please wait for it to finish.")),
    )

    res = client.post(
        "/generate",
        json={"team_id": str(TEAM_ID), "feature_type": "test_tool", "input_params": {}},
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "already have a generation in progress" in res.json()["detail"]


def test_get_job_rejects_non_team_member(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    job = MagicMock(team_id=TEAM_ID)
    db.query.return_value.filter.return_value.first.return_value = job
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: False)

    res = client.get(f"/jobs/{uuid.uuid4()}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 403


def test_get_job_404_when_missing(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    db.query.return_value.filter.return_value.first.return_value = None

    res = client.get(f"/jobs/{uuid.uuid4()}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 404


def test_get_job_returns_status_for_team_member(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    job = MagicMock(
        id=uuid.uuid4(), team_id=TEAM_ID, status="queued",
        output_url=None, error_message=None, credits_charged=5,
    )
    db.query.return_value.filter.return_value.first.return_value = job
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    res = client.get(f"/jobs/{job.id}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 200
    assert res.json()["status"] == "queued"
    assert res.json()["creditsCharged"] == 5


# --------------------------------------------------------------------------- #
# GET /batches/{batch_id}
# --------------------------------------------------------------------------- #

def test_get_batch_returns_status_for_every_job_in_the_batch(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    batch_id = uuid.uuid4()
    j1 = MagicMock(id=uuid.uuid4(), team_id=TEAM_ID, batch_id=batch_id,
                    status="completed", output_url="https://our-storage.test/1.png", error_message=None)
    j2 = MagicMock(id=uuid.uuid4(), team_id=TEAM_ID, batch_id=batch_id,
                    status="queued", output_url=None, error_message=None)
    j3 = MagicMock(id=uuid.uuid4(), team_id=TEAM_ID, batch_id=batch_id,
                    status="failed", output_url=None, error_message="fal reported success but returned no image")
    db.query.return_value.filter.return_value.all.return_value = [j1, j2, j3]
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    res = client.get(f"/batches/{batch_id}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 200
    body = res.json()
    assert body["batchId"] == str(batch_id)
    by_id = {j["jobId"]: j for j in body["jobs"]}
    assert by_id[str(j1.id)]["status"] == "completed"
    assert by_id[str(j1.id)]["outputUrl"] == "https://our-storage.test/1.png"
    assert by_id[str(j2.id)]["status"] == "queued"
    assert by_id[str(j3.id)]["status"] == "failed"
    assert by_id[str(j3.id)]["errorMessage"] == "fal reported success but returned no image"


def test_get_batch_404_when_no_jobs_found(gen_client):
    client, fake_user, db = gen_client
    db.query.return_value.filter.return_value.all.return_value = []

    res = client.get(f"/batches/{uuid.uuid4()}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 404


def test_get_batch_rejects_non_team_member(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    job = MagicMock(id=uuid.uuid4(), team_id=TEAM_ID, status="queued", output_url=None, error_message=None)
    db.query.return_value.filter.return_value.all.return_value = [job]
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: False)

    res = client.get(f"/batches/{uuid.uuid4()}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 403
