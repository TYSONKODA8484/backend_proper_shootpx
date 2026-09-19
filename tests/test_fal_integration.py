"""Real fal.ai integration audit: /generate -> arq enqueue -> submit_generation_to_fal
worker task -> fal.ai -> processing, and the webhook that eventually resolves it.

Covers: enqueue-failure cleanup, duplicate-submission idempotency (arq retries /
double pickup), lock-not-released-on-submit-success, and that FAL_KEY never
leaks into a stored error message.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
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


def _tool(feature_type="test_tool", credit_cost=5, active=True, model_id="fake-model", ai_steps=None):
    return MagicMock(feature_type=feature_type, credit_cost_per_output=credit_cost,
                      is_active=active, fal_model_id=model_id, ai_steps=ai_steps if ai_steps is not None else {})


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
    (or enqueue_job) raising after create_generation_batch already committed
    left credits permanently spent, the job stuck in 'queued' forever, and
    the per-user lock held, with nothing ever going to process it."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5, is_active=True, stage=1)
    monkeypatch.setattr(
        "app.routes.generation.upload_image_to_fal",
        lambda *a, **k: "https://fal.test/uploaded.png",
    )

    job = MagicMock(id=uuid.uuid4(), status="queued", team_id=TEAM_ID, user_id=fake_user.id,
                     credits_charged=5, credits_from_subscription=3, credits_from_topup=2)
    monkeypatch.setattr("app.routes.generation.create_generation_batch", MagicMock(return_value=[job]))
    # fail_and_release counts remaining queued/processing siblings in the
    # batch to decide whether to release the lock; 0 here since this is a
    # single-job batch with nothing else in flight.
    db.query.return_value.filter.return_value.count.return_value = 0

    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)

    async def boom():
        raise ConnectionError("redis unreachable for arq")
    monkeypatch.setattr("app.routes.generation.get_arq_pool", boom)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
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
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.submit_generation_to_fal({}, str(job.id)))
    assert job.status == "processing"
    assert submit.call_count == 1

    # arq retries the SAME job (crash after commit, or a duplicate pickup) --
    # the row now reads 'processing', not 'queued'
    _run(worker.submit_generation_to_fal({}, str(job.id)))

    assert submit.call_count == 1          # fal.ai was NOT called a second time
    refund.assert_not_called()             # and definitely no refund fired


def test_submit_translates_size_and_passes_quality_through_before_sending_to_fal(monkeypatch):
    """The params dict actually sent to fal.ai must carry the translated size
    (image_size preset) -- quality is stored as fal's own real enum value
    directly (auto/low/medium/high) now, so it must reach fal unchanged, not
    translated. job.input_params (read back via GET /jobs and /batches) must
    keep the original user-facing values untouched either way."""
    job = MagicMock(
        id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
        feature_type="test_tool",  # no TOOL_HANDLERS entry -- isolates the translation step itself
        credits_charged=5, credits_from_subscription=3, credits_from_topup=2,
        input_params={"color": "red", "quality": "high", "size": "9:16"},
    )
    original_input_params = dict(job.input_params)
    tool = _tool()
    db = _worker_db(job, tool)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    submit = MagicMock(return_value="fal-req-1")
    monkeypatch.setattr(worker, "submit_to_fal", submit)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    assert job.status == "processing"
    sent_params = submit.call_args.args[1]
    assert sent_params["quality"] == "high"              # already fal's real value -- unchanged
    assert sent_params["image_size"] == "portrait_16_9"  # 9:16 -> portrait_16_9
    assert "size" not in sent_params

    # job.input_params itself is never touched
    assert job.input_params == original_input_params


def test_submit_calls_the_tools_build_instruction_via_a_thread_and_uses_its_result(monkeypatch):
    """The only existing tests using feature_type="test_tool" never exercise
    the TOOL_HANDLERS branch at all (no registry entry for it) -- this covers
    the actual path recolor uses: build_instruction() runs (via
    asyncio.to_thread, since it can make its own blocking HTTP call -- see
    the asyncio.to_thread comments in submit_generation_to_fal) and its
    return value becomes the "prompt" sent to fal.ai."""
    job = MagicMock(id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
                     feature_type="recolor", credits_charged=5,
                     credits_from_subscription=3, credits_from_topup=2,
                     input_params={"color": "red"})
    tool = _tool(feature_type="recolor")
    db = _worker_db(job, tool)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    build_instruction = MagicMock(return_value="Recolor the detected jacket to color red")
    monkeypatch.setattr(worker, "TOOL_HANDLERS", {"recolor": build_instruction})
    submit = MagicMock(return_value="fal-req-1")
    monkeypatch.setattr(worker, "submit_to_fal", submit)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    build_instruction.assert_called_once_with(job, tool, db)
    assert job.status == "processing"
    sent_params = submit.call_args.args[1]
    assert sent_params["prompt"] == "Recolor the detected jacket to color red"
    # Real bug this guards against: tool.ai_steps is an untouched MagicMock
    # here (no ai_steps configured), and MagicMock().get("model") is truthy
    # by default -- before ai_steps["model"] forwarding was scoped to
    # feature_type=="enhance_prompt", this silently injected a spurious
    # "model" key (a MagicMock object) into gpt-image-2/edit's real payload
    # for every tool going through TOOL_HANDLERS, recolor included.
    assert "model" not in sent_params


def test_ai_steps_model_is_forwarded_for_enhance_prompt(monkeypatch):
    """enhance_prompt's real fal model (openrouter/router) needs an explicit
    "model" field to pick the underlying LLM -- tool.ai_steps["model"] must
    still reach the payload for this one tool."""
    job = MagicMock(id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
                     feature_type="enhance_prompt", credits_charged=5,
                     credits_from_subscription=3, credits_from_topup=2,
                     input_params={"prompt": "make it pop", "source_feature_type": "recolor"})
    tool = _tool(feature_type="enhance_prompt")
    tool.ai_steps = {"model": "google/gemini-2.5-flash"}
    db = _worker_db(job, tool)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    build_instruction = MagicMock(return_value="Rewrite this prompt: make it pop")
    monkeypatch.setattr(worker, "TOOL_HANDLERS", {"enhance_prompt": build_instruction})
    submit = MagicMock(return_value="fal-req-1")
    monkeypatch.setattr(worker, "submit_to_fal", submit)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    assert job.status == "processing"
    sent_params = submit.call_args.args[1]
    assert sent_params["model"] == "google/gemini-2.5-flash"


def test_ai_steps_model_is_not_forwarded_for_other_tools_even_if_accidentally_set(monkeypatch):
    """Real landmine this guards against: tool.ai_steps is a raw JSONB column
    hand-edited via SQL with no schema enforcement. gpt-image-2/edit (recolor's
    and creative_photoshoot's real fal model) has NO "model" field in its
    schema -- if a top-level "model" key is ever accidentally added to one of
    THEIR tool_definitions rows (easy mistake: nested steps like
    detect_target/scene_vision already use "model" as a sub-key), it must
    never reach the real generation payload and break every submission for
    that tool."""
    job = MagicMock(id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
                     feature_type="recolor", credits_charged=5,
                     credits_from_subscription=3, credits_from_topup=2,
                     input_params={"color": "red"})
    tool = _tool(feature_type="recolor")
    tool.ai_steps = {"model": "accidentally-set-top-level", "detect_target": {"model": "fal-ai/moondream-next"}}
    db = _worker_db(job, tool)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    build_instruction = MagicMock(return_value="Recolor the jacket to color red")
    monkeypatch.setattr(worker, "TOOL_HANDLERS", {"recolor": build_instruction})
    submit = MagicMock(return_value="fal-req-1")
    monkeypatch.setattr(worker, "submit_to_fal", submit)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    assert job.status == "processing"
    sent_params = submit.call_args.args[1]
    assert "model" not in sent_params


MODEL_SHOOT_GENERATE_MODEL_GENERATION_DEFAULTS = {
    "quality": "low",
    "image_size": {"width": 1024, "height": 1024},
}


def test_model_shoot_generate_model_submits_cleanly_with_zero_input_images(monkeypatch):
    """model_shoot_generate_model's real fal model (openai/gpt-image-2.5/
    flare/text-to-image) takes no input image at all. /generate's generic
    path still always writes input_params["image_urls"] = [] for this tool
    (max_input_images=0 blocks any real upload) -- confirms that flows all
    the way through submit_generation_to_fal cleanly: no image_urls key
    reaches fal, no structured-attribute fields leak through either, and the
    job still reaches 'processing' normally."""
    job = MagicMock(
        id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
        feature_type="model_shoot_generate_model", credits_charged=2,
        credits_from_subscription=2, credits_from_topup=0,
        input_params={
            "gender": "Female", "age_bracket": "Adult (30s-40s)", "ethnicity": "South Asian",
            "skin_tone": "Fair", "body_type": "Slim", "notes": "",
            "image_urls": [],  # always present, always empty for this tool
        },
    )
    tool = _tool(
        feature_type="model_shoot_generate_model", model_id="openai/gpt-image-2.5/flare/text-to-image",
        ai_steps={"generation_defaults": MODEL_SHOOT_GENERATE_MODEL_GENERATION_DEFAULTS},
    )
    db = _worker_db(job, tool)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    build_instruction = MagicMock(return_value="A professional studio portrait of an adult South Asian woman.")
    monkeypatch.setattr(worker, "TOOL_HANDLERS", {"model_shoot_generate_model": build_instruction})
    submit = MagicMock(return_value="fal-req-1")
    monkeypatch.setattr(worker, "submit_to_fal", submit)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    assert job.status == "processing"
    assert submit.call_args.args[0] == "openai/gpt-image-2.5/flare/text-to-image"
    sent_params = submit.call_args.args[1]
    assert sent_params == {
        "prompt": "A professional studio portrait of an adult South Asian woman.",
        "quality": "low",
        "image_size": {"width": 1024, "height": 1024},
    }
    assert "image_urls" not in sent_params
    for field in ("gender", "age_bracket", "ethnicity", "skin_tone", "body_type", "notes"):
        assert field not in sent_params


def test_model_shoot_generate_models_generation_defaults_reach_the_real_fal_payload(monkeypatch):
    """Regression test for the real bug found live: the actual outgoing fal
    request only ever contained {"prompt": ...} -- quality/image_size from
    tool_definitions.ai_steps.generation_defaults never reached it at all,
    so the endpoint silently fell back to its own defaults (quality=high, a
    landscape preset) instead of what's actually configured for this tool.
    Uses the exact ai_steps.generation_defaults shape from the live DB row."""
    job = MagicMock(
        id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
        feature_type="model_shoot_generate_model", credits_charged=2,
        credits_from_subscription=2, credits_from_topup=0,
        input_params={
            "gender": "Male", "age_bracket": "Mature adult (50+)", "ethnicity": "Black",
            "skin_tone": "Deep", "body_type": "Athletic", "notes": "",
            "image_urls": [],
        },
    )
    tool = _tool(
        feature_type="model_shoot_generate_model", model_id="openai/gpt-image-2.5/flare/text-to-image",
        ai_steps={"generation_defaults": MODEL_SHOOT_GENERATE_MODEL_GENERATION_DEFAULTS},
    )
    db = _worker_db(job, tool)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    monkeypatch.setattr(
        worker, "TOOL_HANDLERS",
        {"model_shoot_generate_model": MagicMock(return_value="A studio portrait.")},
    )
    submit = MagicMock(return_value="fal-req-2")
    monkeypatch.setattr(worker, "submit_to_fal", submit)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    sent_params = submit.call_args.args[1]  # the ACTUAL outgoing fal payload, not just that the call didn't error
    assert sent_params["quality"] == "low"
    assert sent_params["image_size"] == {"width": 1024, "height": 1024}
    assert sent_params["prompt"] == "A studio portrait."


def test_model_shoot_generate_models_prompt_wins_if_it_ever_collided_with_a_generation_default_key(monkeypatch):
    """generation_defaults is merged in UNDER params, never over it -- prompt
    (and any future build_instruction-derived key) must always win."""
    job = MagicMock(
        id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
        feature_type="model_shoot_generate_model", credits_charged=2,
        credits_from_subscription=2, credits_from_topup=0,
        input_params={"image_urls": []},
    )
    tool = _tool(
        feature_type="model_shoot_generate_model", model_id="openai/gpt-image-2.5/flare/text-to-image",
        ai_steps={"generation_defaults": {"prompt": "should never win", "quality": "low"}},
    )
    db = _worker_db(job, tool)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "TOOL_HANDLERS", {"model_shoot_generate_model": MagicMock(return_value="real prompt")})
    submit = MagicMock(return_value="fal-req-3")
    monkeypatch.setattr(worker, "submit_to_fal", submit)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    sent_params = submit.call_args.args[1]
    assert sent_params["prompt"] == "real prompt"
    assert sent_params["quality"] == "low"


# --------------------------------------------------------------------------- #
# model_shoot: real bug fixed -- worker.py used to derive image_size from a
# stale aspect-ratio-only preset table (MODEL_SHOOT_ASPECT_RATIO_TO_IMAGE_SIZE),
# which silently ignored "resolution" entirely (every resolution for a given
# aspect ratio got the exact same fixed size). Now uses model_shoot.
# resolve_generation_params, off the real ai_steps.size_map, same pattern as
# listing_photoshoot/creative_photoshoot. model_shoot has no build_instruction
# step (per-job prompt is baked in at job-creation time via per_job_overrides),
# so it goes through the same "else: params = dict(job.input_params)" path as
# listing_photoshoot.
# --------------------------------------------------------------------------- #

MODEL_SHOOT_SIZE_MAP = {
    "1:1": {"1k": {"width": 1920, "height": 1920}, "2k": {"width": 3008, "height": 3008}, "4k": {"width": 4096, "height": 4096}},
    "3:4": {"1k": {"width": 1920, "height": 2560}, "2k": {"width": 2496, "height": 3328}, "4k": {"width": 3072, "height": 4096}},
    "16:9": {"1k": {"width": 3413, "height": 1920}, "2k": {"width": 3755, "height": 2112}, "4k": {"width": 4096, "height": 2304}},
}
MODEL_SHOOT_RESOLUTION_CREDIT = {"1k": 4, "2k": 6, "4k": 8}


@pytest.mark.parametrize("aspect_ratio,resolution", [
    ("1:1", "1k"), ("3:4", "2k"), ("16:9", "4k"),
])
def test_model_shoots_real_image_size_reaches_the_real_fal_payload_for_its_own_resolution(
    monkeypatch, aspect_ratio, resolution,
):
    """Confirms the actual outgoing fal payload's image_size matches
    ai_steps.size_map[aspect_ratio][resolution] exactly -- not the same fixed
    size regardless of which resolution was picked, which is what the old
    stale table did."""
    job = MagicMock(
        id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
        feature_type="model_shoot", credits_charged=6,
        credits_from_subscription=6, credits_from_topup=0,
        input_params={
            "prompt": "A pose prompt already planned via per_job_overrides.",
            "aspect_ratio": aspect_ratio, "resolution": resolution,
            "model_image": "https://fal.test/model.png",
            "garments": [
                {"label": "Top", "image_urls": ["https://fal.test/shirt-front.png", "https://fal.test/shirt-back.png"]},
                {"label": "Bottom", "image_urls": ["https://fal.test/pants.png"]},
            ],
            "reference_images": [],
        },
    )
    tool = _tool(
        feature_type="model_shoot", model_id="fal-ai/bytedance/seedream/v4.5/edit",
        ai_steps={"size_map": MODEL_SHOOT_SIZE_MAP, "resolution_credit": MODEL_SHOOT_RESOLUTION_CREDIT},
    )
    tool.default_output_count = 4
    db = _worker_db(job, tool)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "TOOL_HANDLERS", {})  # model_shoot has no build_instruction step
    submit = MagicMock(return_value="fal-req-model-shoot")
    monkeypatch.setattr(worker, "submit_to_fal", submit)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    assert job.status == "processing"
    sent_params = submit.call_args.args[1]  # the ACTUAL outgoing fal payload
    assert sent_params["image_size"] == MODEL_SHOOT_SIZE_MAP[aspect_ratio][resolution]
    assert sent_params["prompt"] == "A pose prompt already planned via per_job_overrides."
    # Groups flatten in order, images within a group stay in order -- this is
    # the SAME order plan_model_shoot used to label them for the vision call.
    assert sent_params["image_urls"] == [
        "https://fal.test/model.png",
        "https://fal.test/shirt-front.png", "https://fal.test/shirt-back.png",
        "https://fal.test/pants.png",
    ]
    assert "aspect_ratio" not in sent_params
    assert "resolution" not in sent_params
    assert "model_image" not in sent_params
    assert "garments" not in sent_params
    assert "reference_images" not in sent_params


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
    release_slot = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", release)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", release_slot)

    _run(worker.submit_generation_to_fal({}, str(job.id)))
    assert job.status == "failed"
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)
    release.assert_called_once_with(job.user_id)
    release_slot.assert_called_once_with(job.team_id)  # reserved right before the failed submit -- must free it

    # arq retries the same (now-failed) job
    _run(worker.submit_generation_to_fal({}, str(job.id)))

    refund.assert_called_once()        # still just the one call -- NOT double-refunded
    release.assert_called_once()       # NOT released a second time either
    release_slot.assert_called_once()  # NOT released a second time either


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
    release_slot = MagicMock()
    monkeypatch.setattr(worker, "release_generation_lock", release)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", release_slot)

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    assert job.status == "processing"
    assert job.fal_request_id == "fal-req-1"
    release.assert_not_called()
    # the fal slot stays reserved too -- only the webhook (or a sweep timeout)
    # frees it, once the job actually reaches a terminal state
    release_slot.assert_not_called()


def test_unknown_tool_is_a_safe_noop_not_a_crash(monkeypatch):
    job = MagicMock(id=uuid.uuid4(), status="queued", feature_type="ghost_tool")
    db = _worker_db(job, None)  # tool lookup returns nothing
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    submit = MagicMock()
    monkeypatch.setattr(worker, "submit_to_fal", submit)

    _run(worker.submit_generation_to_fal({}, str(job.id)))  # must not raise

    submit.assert_not_called()


# --------------------------------------------------------------------------- #
# fal concurrency limiting: at-capacity re-enqueue, and the real per-team +
# global caps against actual Redis.
# --------------------------------------------------------------------------- #

def _job_created(seconds_ago: float, **overrides):
    now = datetime.now(timezone.utc)
    defaults = dict(
        id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
        feature_type="test_tool", credits_charged=5,
        credits_from_subscription=3, credits_from_topup=2, input_params={},
        created_at=now - timedelta(seconds=seconds_ago),
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


def test_submit_re_enqueues_instead_of_submitting_when_at_capacity(monkeypatch):
    """try_reserve_fal_slot() returning False means fal is at capacity right
    now -- the job must be deferred back onto the queue, NOT sent to fal.ai,
    and it must stay 'queued' (not touched) so a later attempt can pick it up
    cleanly. Freshly created (elapsed ~0s) -> still inside the fast-poll
    window, so the defer stays at 5s."""
    job = _job_created(seconds_ago=0)
    db = _worker_db(job, _tool())
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: False)
    submit = MagicMock()
    monkeypatch.setattr(worker, "submit_to_fal", submit)

    enqueue = MagicMock()
    fake_pool = MagicMock()

    async def fake_call(*a, **k):
        return enqueue(*a, **k)
    fake_pool.enqueue_job = fake_call

    async def fake_get_arq_pool():
        return fake_pool
    monkeypatch.setattr(worker, "get_arq_pool", fake_get_arq_pool)

    _run(worker.submit_generation_to_fal({}, str(job.id)))

    submit.assert_not_called()      # never sent to fal.ai while at capacity
    assert job.status == "queued"   # left alone for the deferred retry to pick up
    enqueue.assert_called_once_with("submit_generation_to_fal", str(job.id), 2, _defer_by=5)  # attempt incremented, fast poll


def test_capacity_retry_gives_up_cleanly_after_the_wall_clock_deadline_instead_of_looping_forever(monkeypatch):
    """Regression test: a real incident where a worker sent many repeated
    requests while investigating this exact loop. pool.enqueue_job() starts a
    brand-new arq job every time (its own try counter reset to 1), so arq's
    own max_tries never bounds this -- without this explicit deadline, a
    permanently-stuck fal concurrency counter would retry forever. This
    proves the deadline actually stops it: refunded, failed, lock released,
    NOT re-enqueued again -- once genuinely past SLOT_ACQUISITION_DEADLINE_
    SECONDS of real wall-clock waiting, not just after N attempts."""
    job = _job_created(seconds_ago=worker.SLOT_ACQUISITION_DEADLINE_SECONDS + 1)
    db = _worker_db(job, _tool())
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: False)  # still at capacity
    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", release)

    monkeypatch.setattr(worker, "get_arq_pool", MagicMock())  # must never even be reached

    _run(worker.submit_generation_to_fal({}, str(job.id), attempt=200))

    worker.get_arq_pool.assert_not_called()  # gave up instead of scheduling yet another retry
    assert job.status == "failed"
    assert job.error_message == worker.GENERIC_START_FAILED_MESSAGE
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)
    release.assert_called_once_with(job.user_id)


def test_capacity_retry_stays_under_the_deadline_keeps_retrying_on_the_slow_poll(monkeypatch):
    """Still inside the deadline (1s of margin left) -- must keep retrying,
    not refund. Also confirms the backoff actually switched to the slow 20s
    poll once past the first minute, instead of hammering Redis with a check
    every 5s for the whole multi-minute wait."""
    job = _job_created(seconds_ago=worker.SLOT_ACQUISITION_DEADLINE_SECONDS - 1)
    db = _worker_db(job, _tool())
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: False)
    refund = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)

    enqueue = MagicMock()
    fake_pool = MagicMock()

    async def fake_call(*a, **k):
        return enqueue(*a, **k)
    fake_pool.enqueue_job = fake_call

    async def fake_get_arq_pool():
        return fake_pool
    monkeypatch.setattr(worker, "get_arq_pool", fake_get_arq_pool)

    _run(worker.submit_generation_to_fal({}, str(job.id), attempt=200))

    enqueue.assert_called_once_with("submit_generation_to_fal", str(job.id), 201, _defer_by=20)  # slow poll
    assert job.status == "queued"
    refund.assert_not_called()


def test_job_surviving_past_the_old_60s_budget_now_succeeds_instead_of_failing(monkeypatch):
    """Direct reproduction of the bug as it existed before this fix: under
    the OLD logic (MAX_CAPACITY_RETRIES=12 * a flat 5s defer =~ 60s), a job
    still waiting for a slot at attempt 13 / ~61s elapsed would have hit
    `attempt >= MAX_CAPACITY_RETRIES` and been failed+refunded right here --
    that generic message, and fal never even touched. Under the new
    wall-clock deadline (900-1200s depending on fal_per_team_concurrency_
    limit), the exact same state must instead keep waiting."""
    job = _job_created(seconds_ago=61)
    db = _worker_db(job, _tool())
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: False)  # still at capacity
    refund = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)

    enqueue = MagicMock()
    fake_pool = MagicMock()

    async def fake_call(*a, **k):
        return enqueue(*a, **k)
    fake_pool.enqueue_job = fake_call

    async def fake_get_arq_pool():
        return fake_pool
    monkeypatch.setattr(worker, "get_arq_pool", fake_get_arq_pool)

    # attempt=13 is exactly what the OLD 5s-fixed-cadence would have reached
    # by ~61s elapsed (12 retries * 5s) -- old code: attempt(13) >= MAX_
    # CAPACITY_RETRIES(12) -> fail_cleanly. New code must not fail here.
    _run(worker.submit_generation_to_fal({}, str(job.id), attempt=13))

    assert job.status == "queued"    # NOT failed -- this is the fix
    refund.assert_not_called()       # credits never touched
    enqueue.assert_called_once_with("submit_generation_to_fal", str(job.id), 14, _defer_by=20)


def test_try_reserve_fal_slot_raising_fails_the_job_cleanly_not_stuck_forever(monkeypatch):
    """The real gap found live: try_reserve_fal_slot() used to sit outside any
    exception handler in submit_generation_to_fal. A transient Redis error
    there would escape uncaught, leaving the job permanently 'queued' with
    credits spent and the lock held -- nothing else would ever clean it up
    (the fal slot itself was never reserved here, so release_fal_slot must
    NOT be called)."""
    job = MagicMock(id=uuid.uuid4(), status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4(),
                     feature_type="test_tool", credits_charged=5,
                     credits_from_subscription=3, credits_from_topup=2, input_params={})
    db = _worker_db(job, _tool())
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    def boom(team_id):
        raise ConnectionError("redis unreachable")
    monkeypatch.setattr(worker, "try_reserve_fal_slot", boom)

    refund = MagicMock()
    release = MagicMock()
    release_slot = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", release)
    monkeypatch.setattr(worker, "release_fal_slot", release_slot)

    _run(worker.submit_generation_to_fal({}, str(job.id)))  # must not raise

    assert job.status == "failed"
    assert job.error_message == worker.GENERIC_START_FAILED_MESSAGE
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)
    release.assert_called_once_with(job.user_id)
    release_slot.assert_not_called()  # no slot was ever actually reserved


def test_re_enqueue_failure_during_capacity_retry_fails_the_job_cleanly(monkeypatch):
    """If even scheduling the deferred retry blows up (arq pool unreachable),
    the job must still end up cleanly failed+refunded, not silently stuck."""
    job = _job_created(seconds_ago=0)
    db = _worker_db(job, _tool())
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: False)

    async def boom():
        raise ConnectionError("arq pool unreachable")
    monkeypatch.setattr(worker, "get_arq_pool", boom)

    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(worker, "refund_credits", refund)
    monkeypatch.setattr(worker, "release_generation_lock", release)

    _run(worker.submit_generation_to_fal({}, str(job.id)))  # must not raise

    assert job.status == "failed"
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)
    release.assert_called_once_with(job.user_id)


def test_fal_slot_cap_real_redis_per_team_only(monkeypatch):
    """Against the real Redis counters (same mechanism as the genlock tests
    above): the per-team cap trips for that team and doesn't block a
    different team, proving the fairness enforcement this cap exists for
    still works. Every reservation this test takes is released again so it
    can't pollute other tests or the real dev counters (see the -14
    global-count corruption this class of test isolation gap caused, found
    live and fixed in fail_and_release/sweep)."""
    from app.core.config import settings
    from app.services.generation_lock import try_reserve_fal_slot, release_fal_slot

    team_a = uuid.uuid4()
    team_b = uuid.uuid4()
    reserved = []

    try:
        # Fill the per-team cap for team_a.
        for _ in range(settings.fal_per_team_concurrency_limit):
            ok = try_reserve_fal_slot(team_a)
            assert ok is True
            reserved.append(team_a)

        # team_a is now at ITS cap -- must be rejected...
        assert try_reserve_fal_slot(team_a) is False
        # ...but team_b is unaffected, proving the per-team cap doesn't starve others.
        assert try_reserve_fal_slot(team_b) is True
        reserved.append(team_b)
    finally:
        for team_id in reserved:
            release_fal_slot(team_id)


def test_fal_slot_reservation_no_longer_gates_on_the_global_account_wide_cap(monkeypatch):
    """Real fix for the repeated-request incident: submitting past fal's own
    account-wide concurrency limit must no longer be pre-checked/rejected on
    our side -- fal's own documented behavior already queues and dispatches
    automatically once that's hit. Proven here by reserving one slot each for
    a generous number of distinct teams (the old global cap config,
    fal_concurrency_limit, is gone entirely now -- there's no config value
    left to size this against), each comfortably under its OWN per-team cap
    -- every single one must still succeed."""
    from app.services.generation_lock import try_reserve_fal_slot, release_fal_slot

    num_teams = 20
    reserved = []
    try:
        for _ in range(num_teams):
            team = uuid.uuid4()  # a fresh team each time -- never approaches the per-team cap
            assert try_reserve_fal_slot(team) is True
            reserved.append(team)
    finally:
        for team_id in reserved:
            release_fal_slot(team_id)


def test_try_reserve_fal_slot_rolls_back_global_increment_if_team_increment_fails(monkeypatch):
    """Real leak found live: try_reserve_fal_slot() increments the global
    counter, THEN the per-team counter. If the team-level incr() ever raises
    (transient Redis error, or a corrupted non-integer value sitting at that
    one key), the global increment was already committed to real Redis --
    without an explicit rollback, it's stuck +1 forever with nothing left to
    ever decrement it."""
    from app.core.cache import redis_client
    from app.services.generation_lock import try_reserve_fal_slot, INFLIGHT_KEY_GLOBAL

    team = uuid.uuid4()
    baseline_global = int(redis_client.get(INFLIGHT_KEY_GLOBAL) or 0)

    real_incr = redis_client.incr

    def flaky_incr(key, *a, **k):
        if key == f"fal:inflight_count:{team}":
            raise ConnectionError("redis blip mid-reservation")
        return real_incr(key, *a, **k)

    monkeypatch.setattr(redis_client, "incr", flaky_incr)

    with pytest.raises(ConnectionError):
        try_reserve_fal_slot(team)

    monkeypatch.undo()  # restore the real incr before reading state back

    assert int(redis_client.get(INFLIGHT_KEY_GLOBAL) or 0) == baseline_global  # rolled back, not leaked
    assert int(redis_client.get(f"fal:inflight_count:{team}") or 0) == 0


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
    monkeypatch.setattr(worker, "try_reserve_fal_slot", lambda team_id: True)
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())

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


# --------------------------------------------------------------------------- #
# reconcile_fal_slots: self-healing against the real Postgres truth for the
# fal in-flight Redis counters (no TTL on these, unlike the generation lock --
# a leaked slot from a killed worker would otherwise never self-correct).
# --------------------------------------------------------------------------- #

def test_reconcile_fal_slots_corrects_undercounted_and_leaked_teams(monkeypatch):
    """Against real Redis (same pattern as the other real-redis tests above):
    a team whose Redis count is stale/wrong gets corrected to the real
    Postgres "processing" count, a team with a pure leak (nonzero Redis,
    zero real processing jobs) gets reset to 0, and the global counter is
    recomputed as the sum -- reproducing exactly the corruption found live
    this session (a per-team counter at 1 with nothing actually processing)."""
    from app.core.cache import redis_client
    from app.services.generation_lock import INFLIGHT_KEY_GLOBAL

    team_a = uuid.uuid4()  # DB truth: 2 jobs processing, Redis stale at 0
    team_b = uuid.uuid4()  # DB truth: 0 jobs processing, Redis leaked at 5
    key_a = f"fal:inflight_count:{team_a}"
    key_b = f"fal:inflight_count:{team_b}"

    baseline_global = redis_client.get(INFLIGHT_KEY_GLOBAL)
    try:
        redis_client.set(key_a, 0)
        redis_client.set(key_b, 5)

        db = MagicMock()
        db.query.return_value.filter.return_value.group_by.return_value.all.return_value = [
            (team_a, 2),
        ]
        monkeypatch.setattr(worker, "SessionLocal", lambda: db)

        _run(worker.reconcile_fal_slots({}))

        assert int(redis_client.get(key_a)) == 2  # corrected up to DB truth
        assert int(redis_client.get(key_b)) == 0  # leak reset to 0
        assert int(redis_client.get(INFLIGHT_KEY_GLOBAL)) == 2  # sum of true counts
    finally:
        redis_client.delete(key_a)
        redis_client.delete(key_b)
        if baseline_global is None:
            redis_client.delete(INFLIGHT_KEY_GLOBAL)
        else:
            redis_client.set(INFLIGHT_KEY_GLOBAL, baseline_global)


def test_reconcile_fal_slots_leaves_an_already_correct_team_untouched(monkeypatch):
    from app.core.cache import redis_client
    from app.services.generation_lock import INFLIGHT_KEY_GLOBAL

    team_a = uuid.uuid4()
    key_a = f"fal:inflight_count:{team_a}"

    baseline_global = redis_client.get(INFLIGHT_KEY_GLOBAL)
    try:
        redis_client.set(key_a, 3)

        db = MagicMock()
        db.query.return_value.filter.return_value.group_by.return_value.all.return_value = [
            (team_a, 3),
        ]
        monkeypatch.setattr(worker, "SessionLocal", lambda: db)

        _run(worker.reconcile_fal_slots({}))

        assert int(redis_client.get(key_a)) == 3
        assert int(redis_client.get(INFLIGHT_KEY_GLOBAL)) == 3
    finally:
        redis_client.delete(key_a)
        if baseline_global is None:
            redis_client.delete(INFLIGHT_KEY_GLOBAL)
        else:
            redis_client.set(INFLIGHT_KEY_GLOBAL, baseline_global)


def test_reconcile_fal_slots_is_registered_on_the_worker(monkeypatch):
    assert worker.reconcile_fal_slots in worker.WorkerSettings.functions
    assert any(
        getattr(job, "coroutine", None) is worker.reconcile_fal_slots
        for job in worker.WorkerSettings.cron_jobs
    )
