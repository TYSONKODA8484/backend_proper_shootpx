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

# release_fal_slot / try_reserve_fal_slot are auto-mocked for every test by
# the autouse fixture in conftest.py (they talk to real Redis, and almost
# nothing here is testing the fal-concurrency-limit feature itself).


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


# --------------------------------------------------------------------------- #
# Orphaned-lock self-healing: a crash/restart between acquire_generation_lock()
# and the job being created (or the except-block release running) leaves the
# lock stranded with nothing backing it. create_generation_batch must clear
# it and proceed ONLY when the lock is old enough to rule out a genuinely
# in-flight request AND there's no real job in the DB behind it.
# --------------------------------------------------------------------------- #

def _multi_model_db(tool, active_job):
    """Distinguishes the ToolDefinition lookup from the GenerationJob
    active-check -- a single db.query(...).filter(...).first() mock can't
    tell these apart since both create_generation_batch's own tool lookup
    and _user_has_active_generation_job's check share that exact shape."""
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "ToolDefinition":
            q.filter.return_value.first.return_value = tool
        elif name == "GenerationJob":
            q.filter.return_value.first.return_value = active_job
        return q

    db.query.side_effect = query
    return db


def test_stale_lock_is_cleared_and_retried_when_old_enough_and_no_active_job(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", MagicMock(side_effect=[False, True]))
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "generation_lock_age_seconds", lambda uid: 45)  # > STALE_LOCK_GRACE_SECONDS
    monkeypatch.setattr(generation_svc, "spend_credits", MagicMock(return_value=(MagicMock(), 5, 0)))
    user_id = uuid.uuid4()
    db = _multi_model_db(_tool(credit_cost=5), active_job=None)  # no active job backing the lock

    jobs = generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {})

    assert len(jobs) == 1  # succeeded -- the stale lock did not block it
    assert released == [user_id]


def test_lock_stays_blocked_when_a_real_active_job_backs_it(monkeypatch):
    """The lock is old enough to otherwise qualify as stale, but a real
    queued/processing job is actually backing it -- must NOT be cleared."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: False)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "generation_lock_age_seconds", lambda uid: 45)
    db = _multi_model_db(_tool(), active_job=MagicMock())  # a real active job exists

    with pytest.raises(ValueError, match="already have a generation in progress"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {})

    assert released == []  # never touched -- a real job is backing this lock


def test_lock_stays_blocked_when_too_recent_even_with_no_active_job(monkeypatch):
    """Protects a genuinely in-flight request: job creation just hasn't
    committed yet, so no DB row exists -- but the lock is too young to
    assume it's orphaned rather than mid-flight."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: False)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "generation_lock_age_seconds", lambda uid: 5)  # well under the grace period
    db = _multi_model_db(_tool(), active_job=None)

    with pytest.raises(ValueError, match="already have a generation in progress"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {})

    assert released == []


def test_lock_stays_blocked_when_age_is_unknown(monkeypatch):
    """generation_lock_age_seconds returns None when the lock key doesn't
    actually exist in Redis (e.g. a race where it expired between the failed
    acquire and this check) -- must fail closed, never self-heal on unknown age."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: False)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "generation_lock_age_seconds", lambda uid: None)
    db = _multi_model_db(_tool(), active_job=None)

    with pytest.raises(ValueError, match="already have a generation in progress"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {})

    assert released == []


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


# --------------------------------------------------------------------------- #
# per_job_overrides -- listing_photoshoot's own path: N jobs, each with its
# own planned prompt merged over the shared input_params, instead of every
# job in the batch sharing one identical input_params dict.
# --------------------------------------------------------------------------- #

def test_per_job_overrides_gives_each_job_its_own_distinct_input_params(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits", MagicMock(return_value=(MagicMock(), 6, 0)))
    db = _job_db(_tool(credit_cost=3))

    overrides = [
        {"prompt": "Hero shot, studio lighting", "shot_type": "hero"},
        {"prompt": "Side angle, studio lighting", "shot_type": "side"},
    ]
    shared_input = {"size": "1:1", "image_urls": ["https://fal.test/product.png"]}

    jobs = generation_svc.create_generation_batch(
        db, TEAM_ID, uuid.uuid4(), "listing_photoshoot", shared_input,
        per_job_overrides=overrides,
    )

    assert len(jobs) == 2
    assert jobs[0].input_params == {**shared_input, **overrides[0]}
    assert jobs[1].input_params == {**shared_input, **overrides[1]}
    assert jobs[0].input_params["prompt"] == "Hero shot, studio lighting"
    assert jobs[1].input_params["prompt"] == "Side angle, studio lighting"
    # the shared fields survive on every job, not just the first
    assert jobs[0].input_params["size"] == "1:1"
    assert jobs[1].input_params["size"] == "1:1"


def test_per_job_overrides_length_wins_even_when_output_count_argument_disagrees(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits", MagicMock(return_value=(MagicMock(), 9, 0)))
    db = _job_db(_tool(credit_cost=3, default_output_count=4))

    overrides = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]

    jobs = generation_svc.create_generation_batch(
        db, TEAM_ID, uuid.uuid4(), "listing_photoshoot", {},
        output_count=1,  # deliberately disagrees with len(overrides)
        per_job_overrides=overrides,
    )

    assert len(jobs) == 3  # overrides length wins, not the output_count argument


def test_per_job_overrides_splits_credits_correctly_across_all_jobs(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits", MagicMock(return_value=(MagicMock(), 7, 2)))
    db = _job_db(_tool(credit_cost=3))

    overrides = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]

    jobs = generation_svc.create_generation_batch(
        db, TEAM_ID, uuid.uuid4(), "listing_photoshoot", {}, per_job_overrides=overrides,
    )

    assert len(jobs) == 3
    assert all(j.credits_charged == 3 for j in jobs)
    # 7 // 3 = 2 each, remainder 1 on the first job; 2 // 3 = 0 each, remainder 2 on the first job
    assert [j.credits_from_subscription for j in jobs] == [3, 2, 2]
    assert [j.credits_from_topup for j in jobs] == [2, 0, 0]
    assert sum(j.credits_from_subscription for j in jobs) == 7
    assert sum(j.credits_from_topup for j in jobs) == 2


def test_without_per_job_overrides_every_job_shares_the_identical_input_params(monkeypatch):
    """Regression guard: every OTHER existing tool's behavior must stay
    exactly as before -- one shared input_params dict, not accidentally
    forked per job."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits", MagicMock(return_value=(MagicMock(), 10, 0)))
    db = _job_db(_tool(credit_cost=5))

    shared_input = {"color": "red", "image_urls": ["https://fal.test/1.png"]}
    jobs = generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "recolor", shared_input, output_count=2)

    assert len(jobs) == 2
    assert jobs[0].input_params == shared_input
    assert jobs[1].input_params == shared_input
    assert jobs[0].input_params is jobs[1].input_params is shared_input  # the exact same object, not a copy


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


def test_misconfigured_zero_default_output_count_raises_a_clean_value_error(monkeypatch):
    """Regression: output_count is later used as a divisor for the
    credit-pool split. A tool_definitions row hand-edited via SQL to
    default_output_count=0 (or negative) used to reach that division
    directly and crash with an unhandled ZeroDivisionError -- which isn't a
    ValueError, so /generate's `except ValueError` never caught it and the
    request 500'd instead of failing cleanly. Must raise ValueError instead,
    same as every other "this request can't be fulfilled" case here."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    spend = MagicMock(return_value=(MagicMock(), 0, 0))
    monkeypatch.setattr(generation_svc, "spend_credits", spend)
    db = _job_db(_tool(default_output_count=0))

    with pytest.raises(ValueError, match="output_count must be between 1"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {}, output_count=0)

    spend.assert_not_called()  # rejected before any credits are touched


def test_output_count_over_the_max_raises_a_clean_value_error_even_from_a_tool_default(monkeypatch):
    """The MAX_OUTPUT_COUNT cap (see app/routes/generation.py) previously only
    guarded a caller-supplied output_count -- a tool_definitions row whose
    default_output_count exceeds it (DB-only misconfiguration, but nothing
    enforced it) could still create an oversized batch. Now enforced here too,
    on the resolved value regardless of where it came from."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    spend = MagicMock()
    monkeypatch.setattr(generation_svc, "spend_credits", spend)
    db = _job_db(_tool(default_output_count=generation_svc.MAX_OUTPUT_COUNT + 1))

    with pytest.raises(ValueError, match="output_count must be between 1"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {}, output_count=0)

    spend.assert_not_called()


# --------------------------------------------------------------------------- #
# _resolve_credit_cost — per-tier pricing (Standard/Advanced/Premium selects)
# --------------------------------------------------------------------------- #

QUALITY_TIER_SCHEMA = [
    {
        "name": "quality",
        "type": "select",
        "label": "Quality",
        "options": [
            {"value": "standard", "credit_cost": 5},
            {"value": "advanced", "credit_cost": 10},
            {"value": "premium", "credit_cost": 20},
        ],
    },
]


@pytest.mark.parametrize("quality,expected_cost", [
    ("standard", 5),
    ("advanced", 10),
    ("premium", 20),
])
def test_resolve_credit_cost_picks_the_selected_tier(quality, expected_cost):
    cost = generation_svc._resolve_credit_cost(QUALITY_TIER_SCHEMA, {"quality": quality}, fallback_cost=1)
    assert cost == expected_cost


def test_resolve_credit_cost_falls_back_when_selected_value_matches_no_tier():
    cost = generation_svc._resolve_credit_cost(QUALITY_TIER_SCHEMA, {"quality": "unknown-tier"}, fallback_cost=7)
    assert cost == 7


def test_resolve_credit_cost_ignores_plain_string_options_without_credit_cost():
    """A select field whose options are plain strings (e.g. size/aspect ratio)
    must never be mistaken for a priced tier -- only options shaped as dicts
    with a credit_cost key opt into per-tier pricing."""
    schema = [{"name": "size", "type": "select", "options": ["1:1", "16:9"]}]
    cost = generation_svc._resolve_credit_cost(schema, {"size": "16:9"}, fallback_cost=3)
    assert cost == 3


def test_create_batch_charges_the_selected_tiers_credit_cost_not_the_flat_default(monkeypatch):
    """End-to-end through create_generation_batch: the tool's flat
    credit_cost_per_output must be overridden by the selected quality tier's
    own credit_cost, and that per-tier cost is what's split across every job
    in a multi-output batch."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    spend = MagicMock(return_value=(MagicMock(), 40, 0))
    monkeypatch.setattr(generation_svc, "spend_credits", spend)
    tool = _tool(credit_cost=5)
    tool.param_schema = QUALITY_TIER_SCHEMA
    db = _job_db(tool)

    jobs = generation_svc.create_generation_batch(
        db, TEAM_ID, uuid.uuid4(), "test_tool", {"quality": "premium"}, output_count=2,
    )

    assert len(jobs) == 2
    spend.assert_called_once_with(db, TEAM_ID, 40, commit=False)  # 20 (premium) * 2, not 5 (flat) * 2
    assert all(j.credits_charged == 20 for j in jobs)


# --------------------------------------------------------------------------- #
# handle_fal_webhook — success/failure/refund-split/idempotent replay
# --------------------------------------------------------------------------- #

def _job(status="queued", credits_charged=5, from_sub=3, from_topup=2, batch_id=None, fal_request_id=None):
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
        fal_request_id=fal_request_id,
    )


def _webhook_db(job, remaining_in_batch=0):
    """remaining_in_batch: how many OTHER jobs in the same batch are still
    queued/processing -- 0 or 1 means "this was the last one" (the count
    includes this job's own not-yet-flushed row, see fail_and_release /
    handle_fal_webhook's `remaining <= 1` check), anything higher means
    siblings are still in flight and the lock must stay held."""
    db = MagicMock()
    (db.query.return_value.filter.return_value.populate_existing.return_value
       .with_for_update.return_value.first.return_value) = job
    db.query.return_value.filter.return_value.count.return_value = remaining_in_batch
    return db


# --------------------------------------------------------------------------- #
# request_id correlation: a genuinely fal-signed webhook for a DIFFERENT
# request must never be applied to this job (replay / cross-job substitution).
# --------------------------------------------------------------------------- #

def test_webhook_with_mismatched_request_id_is_ignored(monkeypatch):
    """Real gap found live: ?job_id=... isn't part of what fal signs, so
    nothing previously stopped a genuinely fal-signed webhook for one request
    being replayed onto a different job_id. Must be rejected before touching
    status/credits/lock at all."""
    job = _job(status="processing", fal_request_id="fal-req-REAL")
    db = _webhook_db(job)
    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "request_id": "fal-req-ATTACKERS-OWN",
         "payload": {"images": [{"url": "https://attacker.test/x.png"}]}},
    )

    assert job.status == "processing"  # untouched
    refund.assert_not_called()
    release.assert_not_called()
    db.commit.assert_not_called()


def test_webhook_with_matching_request_id_still_works(monkeypatch):
    """Regression check: the correlation check must not break the normal path."""
    job = _job(status="processing", fal_request_id="fal-req-REAL")
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage",
                         MagicMock(return_value="https://our-storage.test/ok.png"))

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "request_id": "fal-req-REAL",
         "payload": {"images": [{"url": "https://fal.test/ok.png"}]}},
    )

    assert job.status == "completed"


def test_webhook_without_a_request_id_field_is_not_rejected(monkeypatch):
    """Our own internal delivery (worker.check_generation_timeouts resolving
    a job via fal's real status API, not an actual HTTP webhook) always sets
    request_id correctly -- but stay permissive if it's ever absent rather
    than fail closed on a missing field that isn't itself suspicious."""
    job = _job(status="processing", fal_request_id="fal-req-REAL")
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage",
                         MagicMock(return_value="https://our-storage.test/ok.png"))

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"images": [{"url": "https://fal.test/ok.png"}]}},
    )

    assert job.status == "completed"


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


# --------------------------------------------------------------------------- #
# enhance_prompt webhook branch -- text output, not an image download/upload
# --------------------------------------------------------------------------- #

def test_webhook_success_sets_output_text_for_enhance_prompt_without_touching_output_url(monkeypatch):
    job = _job()
    job.feature_type = "enhance_prompt"
    job.output_text = None
    db = _webhook_db(job)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    download = MagicMock()
    upload = MagicMock()
    monkeypatch.setattr(generation_svc, "download_from_url", download)
    monkeypatch.setattr(generation_svc, "upload_to_storage", upload)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"output": "A vivid, richly detailed enhanced prompt."}},
    )

    assert job.status == "completed"
    assert job.output_text == "A vivid, richly detailed enhanced prompt."
    assert job.output_url is None
    download.assert_not_called()
    upload.assert_not_called()
    refund.assert_not_called()
    assert released == [job.user_id]
    db.commit.assert_called_once()


def test_webhook_success_with_no_text_fails_and_refunds_enhance_prompt(monkeypatch):
    job = _job()
    job.feature_type = "enhance_prompt"
    job.output_text = None
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {}},
    )

    assert job.status == "failed"
    assert job.output_text is None
    assert job.error_message == "fal reported success but returned no text"
    refund.assert_called_once_with(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)


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
    (db.query.return_value.filter.return_value.populate_existing.return_value
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
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 403


def test_generate_allows_team_member_and_surfaces_value_error_as_400(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5)
    monkeypatch.setattr(
        "app.routes.generation.upload_image_to_fal",
        lambda *a, **k: "https://fal.test/uploaded.png",
    )
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        MagicMock(side_effect=ValueError("You already have a generation in progress. Please wait for it to finish.")),
    )

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "already have a generation in progress" in res.json()["detail"]


# --------------------------------------------------------------------------- #
# /generate: multi-image upload + max_input_images enforcement
# --------------------------------------------------------------------------- #

def test_generate_rejects_unknown_tool_before_any_upload_or_charge(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = None  # tool lookup finds nothing
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "ghost_tool"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert res.json()["detail"] == "Unknown tool"
    upload.assert_not_called()  # rejected before ever touching fal's CDN


def test_generate_rejects_more_images_than_the_tools_max_input_images(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=1)
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[
            ("images", ("one.png", b"one", "image/png")),
            ("images", ("two.png", b"two", "image/png")),
        ],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "at most 1" in res.json()["detail"]
    upload.assert_not_called()  # rejected before uploading anything to fal's CDN


# --------------------------------------------------------------------------- #
# /generate: schema validation + image checks must happen BEFORE any fal
# upload cost is paid -- a request that's going to be rejected anyway
# shouldn't burn real fal.ai upload quota every time.
# --------------------------------------------------------------------------- #

def test_generate_rejects_output_count_over_the_cap_before_any_upload(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5, param_schema=[])
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool", "output_count": "9999"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "output_count" in res.json()["detail"]
    upload.assert_not_called()


def test_generate_omitted_output_count_does_not_reject_with_500(gen_client, monkeypatch):
    """output_count left off the request entirely must not crash the
    MAX_OUTPUT_COUNT comparison (None > int) -- regression guard for the
    Form(1) -> Form(None) change that made tool.default_output_count
    actually reachable."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5, param_schema=[])
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/x.png"))
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        MagicMock(return_value=[MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")]),
    )

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},  # no output_count at all
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200


def test_generate_omitting_output_count_lets_listing_photoshoot_reach_its_own_default(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, default_output_count=4, param_schema=[],
    )
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/x.png"))
    plan = MagicMock(return_value=[{"shot_type": "hero", "prompt": "x"}] * 4)
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", plan)
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        MagicMock(return_value=[MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued") for _ in range(4)]),
    )

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "listing_photoshoot"},  # no output_count at all
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    plan.assert_called_once_with(db.query.return_value.filter.return_value.first.return_value, ["https://fal.test/x.png"], None, 4)


def test_generate_rejects_missing_required_field_before_any_upload(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5,
        param_schema=[{"name": "color", "type": "color", "label": "Color", "required": True}],
    )
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "recolor"},  # no "color" field sent
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "Missing required field: Color" in res.json()["detail"]
    upload.assert_not_called()  # rejected before ever touching fal's CDN


def test_generate_rejects_disallowed_image_content_type_before_any_upload(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5, param_schema=[])
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[("images", ("payload.exe", b"not-an-image", "application/x-msdownload"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "Unsupported image type" in res.json()["detail"]
    upload.assert_not_called()


def test_generate_rejects_oversized_image_before_any_upload(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5, param_schema=[])
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)
    monkeypatch.setattr("app.routes.generation.MAX_IMAGE_SIZE_BYTES", 10)  # tiny, for the test

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[("images", ("big.png", b"way more than ten bytes of data", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "MB limit" in res.json()["detail"]
    upload.assert_not_called()


def test_generate_uploads_every_image_and_passes_all_urls_through(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=3)

    upload = MagicMock(side_effect=["https://fal.test/1.png", "https://fal.test/2.png"])
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count):
        captured["input_params"] = input_params
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[
            ("images", ("one.png", b"one", "image/png")),
            ("images", ("two.png", b"two", "image/png")),
        ],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    assert upload.call_count == 2
    # image_urls is ALWAYS a list, in upload order, one URL per uploaded file
    assert captured["input_params"]["image_urls"] == ["https://fal.test/1.png", "https://fal.test/2.png"]


def test_generate_forwards_idea_field_for_creative_photoshoot(gen_client, monkeypatch):
    """creative_photoshoot needs "idea" -- not one of Recolor's fields --
    forwarded through to create_generation_batch, same class of gap that
    previously blocked enhance_prompt's "prompt"/"source_feature_type"."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5,
        param_schema=[
            {"name": "idea", "type": "select", "label": "Idea", "options": ["Sci-Fi", "Luxury"], "required": False},
            {"name": "prompt", "type": "text", "label": "Prompt", "required": False},
        ],
    )
    upload = MagicMock(return_value="https://fal.test/uploaded.png")
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count):
        captured["input_params"] = input_params
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "creative_photoshoot", "idea": "Sci-Fi"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    assert captured["input_params"]["idea"] == "Sci-Fi"


def test_generate_plans_shots_and_uses_per_job_overrides_for_listing_photoshoot(gen_client, monkeypatch):
    """listing_photoshoot is the one tool where /generate must NOT call
    create_generation_batch with a single shared input_params -- it plans N
    distinct shots first and passes per_job_overrides instead."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    tool = MagicMock(
        max_input_images=5, default_output_count=4,
        param_schema=[{"name": "prompt", "type": "text", "label": "Prompt", "required": False}],
    )
    db.query.return_value.filter.return_value.first.return_value = tool
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/uploaded.png"))

    planned_shots = [
        {"shot_type": "hero", "prompt": "Hero shot, studio lighting"},
        {"shot_type": "side", "prompt": "Side angle, studio lighting"},
    ]
    plan = MagicMock(return_value=planned_shots)
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", plan)

    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count=1, per_job_overrides=None, lock_already_held=False):
        captured["output_count"] = output_count
        captured["per_job_overrides"] = per_job_overrides
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued") for _ in (per_job_overrides or [None])]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "listing_photoshoot"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    plan.assert_called_once_with(tool, ["https://fal.test/uploaded.png"], None, 4)  # falls back to tool.default_output_count
    assert captured["per_job_overrides"] == planned_shots
    assert captured["output_count"] == 4


def test_generate_never_plans_shots_when_a_generation_is_already_in_progress(gen_client, monkeypatch):
    """Real bug found live: the lock used to be checked INSIDE
    create_generation_batch, which only runs AFTER plan_listing_shots -- a
    request that was going to be rejected anyway still burned a real,
    billable fal vision call for nothing. The lock must be checked first."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, default_output_count=4, param_schema=[],
    )
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/x.png"))
    monkeypatch.setattr("app.routes.generation.acquire_or_heal_generation_lock", lambda db, uid: False)
    plan = MagicMock()
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", plan)
    create_batch = MagicMock()
    monkeypatch.setattr("app.routes.generation.create_generation_batch", create_batch)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "listing_photoshoot"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "already have a generation in progress" in res.json()["detail"]
    plan.assert_not_called()  # the whole point -- never burn the billable call
    create_batch.assert_not_called()


def test_generate_releases_the_lock_itself_when_shot_planning_fails_after_acquiring_it(gen_client, monkeypatch):
    """The route acquires the lock itself before planning (lock_already_held
    passes ownership to create_generation_batch only on success) -- if
    planning fails, the route must release it, or every subsequent request
    from this user would be wrongly blocked until the lock's TTL expires."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, default_output_count=4, param_schema=[],
    )
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/x.png"))
    monkeypatch.setattr("app.routes.generation.acquire_or_heal_generation_lock", lambda db, uid: True)
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", MagicMock(side_effect=RuntimeError("fal is down")))
    release = MagicMock()
    monkeypatch.setattr("app.routes.generation.release_generation_lock", release)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "listing_photoshoot"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 503
    release.assert_called_once_with(fake_user.id)


def test_generate_surfaces_a_clean_503_when_shot_planning_fails(gen_client, monkeypatch):
    """The vision call is a real, billable external request -- a raw
    exception must not leak as an unhandled 500."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, default_output_count=4, param_schema=[],
    )
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/uploaded.png"))
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", MagicMock(side_effect=RuntimeError("fal is down")))
    create_batch = MagicMock()
    monkeypatch.setattr("app.routes.generation.create_generation_batch", create_batch)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "listing_photoshoot"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 503
    create_batch.assert_not_called()  # never charged/created anything after the planning failure


def test_generate_accepts_enhance_prompt_with_no_images(gen_client, monkeypatch):
    """enhance_prompt has max_input_images=0 and needs "prompt" +
    "source_feature_type" instead of any of Recolor's fields -- confirms the
    route actually accepts both, requires no file upload, and forwards them
    through to create_generation_batch untouched."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=0,
        param_schema=[
            {"name": "prompt", "type": "text", "label": "Prompt", "required": True},
            {"name": "source_feature_type", "type": "text", "label": "Source tool", "required": True},
        ],
    )
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count):
        captured["input_params"] = input_params
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={
            "team_id": str(TEAM_ID), "feature_type": "enhance_prompt",
            "prompt": "a cool sneaker", "source_feature_type": "recolor",
        },
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    upload.assert_not_called()  # no images -- must never try to upload anything
    assert captured["input_params"]["prompt"] == "a cool sneaker"
    assert captured["input_params"]["source_feature_type"] == "recolor"
    assert captured["input_params"]["image_urls"] == []


def test_get_job_returns_output_text_for_enhance_prompt(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    job = MagicMock(
        id=uuid.uuid4(), team_id=TEAM_ID, status="completed",
        output_url=None, output_text="an enhanced, more vivid prompt",
        error_message=None, credits_charged=0,
    )
    db.query.return_value.filter.return_value.first.return_value = job
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    res = client.get(f"/jobs/{job.id}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 200
    body = res.json()
    assert body["outputText"] == "an enhanced, more vivid prompt"
    assert body["outputUrl"] is None


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
