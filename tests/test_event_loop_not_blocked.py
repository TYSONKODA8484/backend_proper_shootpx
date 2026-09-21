"""Blocking outbound calls in async handlers must not freeze the whole API.

/generate's planner calls and POST /webhooks/fal's download+upload are
blocking HTTP calls, and with retries (3 attempts, up to ~94s worst case) a
stall would hold the server's single event loop that long -- freezing EVERY
other user's request, not just this one. They run in a thread instead.

Each test starts a deliberately slow blocking call and, while it is running,
checks the event loop is still free (see _loop_lag_while).
"""
import asyncio
import time
import uuid
from unittest.mock import MagicMock

import httpx
import pytest

from app.core.database import get_db
from app.deps import get_current_user
from app.main import app
from app.routes import generation as gen

SLOW = 0.8            # how long the fake blocking call holds its thread
MAX_LOOP_LAG = 0.3     # a free event loop wakes a timer within a few ms


@pytest.fixture(autouse=True)
def _overrides():
    app.dependency_overrides[get_current_user] = lambda: MagicMock(id=uuid.uuid4())
    app.dependency_overrides[get_db] = lambda: MagicMock()
    yield
    app.dependency_overrides.clear()


async def _loop_lag_while(slow_request):
    """Start `slow_request`, then measure how LATE the event loop wakes a
    0.15s timer while it runs.

    Measuring lag (not the latency of a request issued afterwards) is what
    actually detects a blocked loop: if the slow call blocks the loop, this
    coroutine cannot resume until the block ends, so the timer fires ~SLOW
    seconds late. Timing a /health request from AFTER the resume would look
    healthy even with the loop frozen -- the first version of this test had
    exactly that flaw and passed against the broken code.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        slow_task = asyncio.create_task(slow_request(client))
        started = time.perf_counter()
        await asyncio.sleep(0.15)
        lag = time.perf_counter() - started - 0.15
        health = await client.get("/health")
        slow_response = await slow_task
        return health, lag, slow_response


def _slow(result):
    def fn(*a, **k):
        time.sleep(SLOW)                           # a blocking call, like a stalled HTTP request
        return result
    return fn


def test_the_fal_webhook_download_and_upload_do_not_block_other_requests(monkeypatch):
    handled = []
    monkeypatch.setattr(gen, "verify_fal_webhook", lambda headers, body: None)
    monkeypatch.setattr(gen, "handle_fal_webhook", lambda db, job_id, payload: (time.sleep(SLOW), handled.append(job_id)))

    async def slow(client):
        return await client.post(f"/webhooks/fal?job_id={uuid.uuid4()}", json={"status": "OK"})

    health, lag, webhook = asyncio.run(_loop_lag_while(slow))

    assert health.status_code == 200
    assert lag < MAX_LOOP_LAG, f"event loop woke {lag:.2f}s late -- it was blocked"
    assert webhook.status_code == 200 and len(handled) == 1     # and the work itself still ran


def _setup_generate(monkeypatch, tool, resolved=None):
    monkeypatch.setattr(gen, "is_team_member", lambda db, tid, uid: True)
    monkeypatch.setattr(gen, "get_tool_definition", lambda db, ft: tool)
    monkeypatch.setattr(gen, "acquire_or_heal_generation_lock", lambda db, uid: True)
    monkeypatch.setattr(gen, "upload_image_to_fal", lambda *a: "https://fal.test/x.png")
    monkeypatch.setattr(gen, "_resolve_source_job_urls", lambda db, tid, ids: resolved or {})
    monkeypatch.setattr(
        gen, "create_generation_batch",
        lambda *a, **k: [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")],
    )

    async def fake_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr(gen, "get_arq_pool", fake_pool)


def test_listing_shot_planning_does_not_block_other_requests(monkeypatch):
    tool = MagicMock(max_input_images=5, default_output_count=4, is_active=True, stage=1, param_schema=[])
    _setup_generate(monkeypatch, tool)
    shots = [{"shot_type": "front", "prompt": "front view"}] * 4
    monkeypatch.setattr(gen, "plan_listing_shots", _slow(shots))

    async def slow(client):
        return await client.post(
            "/generate",
            data={"team_id": str(uuid.uuid4()), "feature_type": "listing_photoshoot"},
            files=[("images", ("p.png", b"img", "image/png"))],
        )

    health, lag, generate = asyncio.run(_loop_lag_while(slow))

    assert lag < MAX_LOOP_LAG, f"event loop woke {lag:.2f}s late -- planning blocked it"
    assert generate.status_code == 200


def test_model_shoot_planning_does_not_block_other_requests(monkeypatch):
    model_job, top_job = uuid.uuid4(), uuid.uuid4()
    tool = MagicMock(max_input_images=10, default_output_count=2, is_active=True, stage=1, param_schema=[])
    _setup_generate(monkeypatch, tool, {model_job: "https://s.test/m.png", top_job: "https://s.test/t.png"})
    monkeypatch.setattr(gen, "plan_model_shoot", _slow({"blocked": False, "reason": "", "prompts": ["pose 1", "pose 2"]}))

    async def slow(client):
        return await client.post(
            "/generate",
            data={
                "team_id": str(uuid.uuid4()), "feature_type": "model_shoot",
                "model_source_job_id": str(model_job), "top_source_job_ids": [str(top_job)],
            },
        )

    health, lag, generate = asyncio.run(_loop_lag_while(slow))

    assert lag < MAX_LOOP_LAG, f"event loop woke {lag:.2f}s late -- planning blocked it"
    assert generate.status_code == 200
