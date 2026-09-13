"""Real fal.ai integration audit: /generate -> arq enqueue -> submit_generation_to_fal
worker task -> fal.ai -> processing, and the webhook that eventually resolves it.

Covers: enqueue-failure cleanup, duplicate-submission idempotency (arq retries /
double pickup), lock-not-released-on-submit-success, and that FAL_KEY never
leaks into a stored error message.
"""

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.database import get_db
from app.core.limiter import limiter
from app.deps import get_current_user
from app.main import app
from app.services import generation as generation_svc
from app.services.generation_lock import acquire_generation_lock, release_generation_lock
from app import worker

limiter.enabled = False

TEAM_ID = uuid.uuid4()


def _tool(feature_type="test_tool", credit_cost=5, active=True, model_id="fake-model"):
    return MagicMock(feature_type=feature_type, credit_cost_per_output=credit_cost,
                      is_active=active, fal_model_id=model_id)


# --------------------------------------------------------------------------- #
# /generate: arq enqueue failure must not leave a stuck job + spent credits +
# a held lock that nothing will ever release.
# --------------------------------------------------------------------------- #

@pytest.fixture
def gen_client():
    fake_user = MagicMock(id=uuid.uuid4())
    db = MagicMock()
    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app, raise_server_exceptions=False), fake_user, db
    app.dependency_overrides.clear()


def test_generate_cleans_up_when_arq_enqueue_fails(gen_client, monkeypatch):
    """Reproduces the real bug found live against the dev DB: get_arq_pool()
    (or enqueue_job) raising after create_generation_job already committed
    left credits permanently spent, the job stuck in 'queued' forever, and
    the per-user lock held, with nothing ever going to process it."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    job = MagicMock(id=uuid.uuid4(), status="queued", team_id=TEAM_ID, user_id=fake_user.id,
                     credits_charged=5, credits_from_subscription=3, credits_from_topup=2)
    monkeypatch.setattr("app.routes.generation.create_generation_job", MagicMock(return_value=job))

    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)

    async def boom():
        raise ConnectionError("redis unreachable for arq")
    monkeypatch.setattr("app.routes.generation.get_arq_pool", boom)

    res = client.post(
        "/generate",
        json={"team_id": str(TEAM_ID), "feature_type": "test_tool", "input_params": {}},
        headers={"Authorization": "Bearer x"},
    )

    # must be a clean, non-500 error -- not a raw unhandled exception
    assert res.status_code in (500, 503)
    if res.status_code == 500:
        pytest.fail("enqueue failure must be caught and cleaned up, not surfaced as a raw 500")

    # the job must be failed, not left stuck in 'queued' forever
    assert job.status == "failed"
    # credits must be refunded, in the exact split they were charged
    refund.assert_called_once_with(db, TEAM_ID, 5, 3, 2)
    # the lock must be released -- nothing will ever process this job otherwise
    release.assert_called_once_with(fake_user.id)


# --------------------------------------------------------------------------- #
# submit_generation_to_fal: duplicate run (arq retry / double pickup) must not
# double-submit to fal.ai or double-refund.
# --------------------------------------------------------------------------- #

def _run(coro):
    return asyncio.run(coro)


def _worker_db(job, tool):
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "GenerationJob":
            q.filter.return_value.with_for_update.return_value.first.return_value = job
        elif name == "ToolDefinition":
            q.filter.return_value.first.return_value = tool
        return q

    db.query.side_effect = query
    return db


def test_duplicate_run_after_success_does_not_resubmit(monkeypatch):
    job = MagicMock(id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
                     feature_type="test_tool", credits_charged=5,
                     credits_from_subscription=3, credits_from_topup=2, input_params={})
    tool = _tool()
    db = _worker_db(job, tool)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    submit = MagicMock(return_value="fal-req-1")
    monkeypatch.setattr(worker, "submit_to_fal", submit)
    refund = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)

    _run(worker.submit_generation_to_fal({}, str(job.id)))
    assert job.status == "processing"
    assert submit.call_count == 1

    # arq retries the SAME job (crash after commit, or a duplicate pickup) --
    # the row now reads 'processing', not 'queued'
    _run(worker.submit_generation_to_fal({}, str(job.id)))

    assert submit.call_count == 1          # fal.ai was NOT called a second time
    refund.assert_not_called()             # and definitely no refund fired


def test_duplicate_run_after_failure_does_not_double_refund(monkeypatch):
    job = MagicMock(id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
                     feature_type="test_tool", credits_charged=5,
                     credits_from_subscription=3, credits_from_topup=2, input_params={})
    tool = _tool()
    db = _worker_db(job, tool)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    monkeypatch.setattr(worker, "submit_to_fal", MagicMock(side_effect=RuntimeError("fal 500")))
    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", release)

    _run(worker.submit_generation_to_fal({}, str(job.id)))
    assert job.status == "failed"
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)
    release.assert_called_once_with(job.user_id)

    # arq retries the same (now-failed) job
    _run(worker.submit_generation_to_fal({}, str(job.id)))

    refund.assert_called_once()   # still just the one call -- NOT double-refunded
    release.assert_called_once()  # NOT released a second time either


def test_lock_is_not_released_on_successful_fal_submit(monkeypatch):
    """The lock must only be released by the real webhook later -- releasing
    it here would let the user submit a second generation before the first
    one's actual result comes back."""
    job = MagicMock(id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
                     feature_type="test_tool", credits_charged=5,
                     credits_from_subscription=3, credits_from_topup=2, input_params={})
    db = _worker_db(job, _tool())
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "submit_to_fal", MagicMock(return_value="fal-req-1"))
    release = MagicMock()
    monkeypatch.setattr(worker, "release_generation_lock", release)

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    assert job.status == "processing"
    assert job.fal_request_id == "fal-req-1"
    release.assert_not_called()


def test_unknown_tool_is_a_safe_noop_not_a_crash(monkeypatch):
    job = MagicMock(id=uuid.uuid4(), status="queued", feature_type="ghost_tool")
    db = _worker_db(job, None)  # tool lookup returns nothing
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    submit = MagicMock()
    monkeypatch.setattr(worker, "submit_to_fal", submit)

    _run(worker.submit_generation_to_fal({}, str(job.id)))  # must not raise

    submit.assert_not_called()


# --------------------------------------------------------------------------- #
# FAL_KEY must never leak into a stored/returned error message.
# --------------------------------------------------------------------------- #

def test_fal_key_never_appears_in_worker_error_message(monkeypatch):
    job = MagicMock(id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
                     feature_type="test_tool", credits_charged=5,
                     credits_from_subscription=3, credits_from_topup=2, input_params={})
    db = _worker_db(job, _tool())
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "refund_credits", MagicMock())
    monkeypatch.setattr(worker, "release_generation_lock", MagicMock())

    # Simulate the worst case: an exception whose message happens to include
    # the raw Authorization header value, exactly as a raw httpx request-prep
    # error theoretically could.
    monkeypatch.setattr(
        worker, "submit_to_fal",
        MagicMock(side_effect=RuntimeError(f"request failed, headers were: Authorization: Key {settings.fal_key}")),
    )

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    assert settings.fal_key not in job.error_message, (
        "FAL_KEY leaked into job.error_message, which /jobs/{id} returns to any team member"
    )
