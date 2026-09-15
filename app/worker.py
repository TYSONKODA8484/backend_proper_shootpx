import asyncio
import logging
logging.basicConfig(level=logging.INFO)

from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from arq.connections import RedisSettings
from arq.cron import cron

from app.core.config import settings
from app.core.cache import redis_client
from app.core.database import SessionLocal
from app.core.fal_client import submit_to_fal, check_fal_status, fetch_fal_result
from app.core.arq_pool import get_arq_pool
import app.models  # noqa: F401 -- registers every model on Base.metadata before any query runs (see app/models/__init__.py)
from app.models.team_subscription import TeamSubscription
from app.models.subscription import Subscription
from app.models.team import Team
from app.models.generation_job import GenerationJob
from app.services.tool_definitions import get_tool_definition
from app.services.credits import refill_subscription_credits, refund_credits
from app.services.generation_lock import (
    release_generation_lock, try_reserve_fal_slot, release_fal_slot, INFLIGHT_KEY_GLOBAL,
)
from app.services import generation as generation_svc
from app.services.generation import TIMEOUT_MESSAGE, SWEEP_TIMEOUT_MESSAGE
from app.tools.registry import TOOL_HANDLERS

logger = logging.getLogger(__name__)

# Recolor's param_schema now stores "quality" as fal's own real values
# (auto/low/medium/high, matching openai/gpt-image-2/edit's actual accepted
# vocabulary directly) -- no translation needed for that field any more. Size
# is still our own plain aspect-ratio strings ("9:16", etc.), which DO need
# translating: fal's `image_size` only accepts a fixed set of named presets
# (or an explicit {width, height} object) -- confirmed against
# https://fal.ai/models/openai/gpt-image-2/edit/api. This table is currently
# written against exactly that model's vocabulary; if a future tool reuses
# the "size" field name with different values against a different fal model,
# this translation would need to become per-tool rather than global.

# fal has no named preset for a plain 2:3 / 3:2 ratio, so those fall back to
# an explicit {width, height} object. Constraints (per the docs above): both
# dimensions multiples of 16, max edge 3840px, aspect ratio <= 3:1, total
# pixels between 655,360 and 8,294,400 -- 1024x1536 (and its 1536x1024 swap)
# satisfy all four comfortably.
FAL_SIZE_TRANSLATION = {
    "original": "auto",
    "1:1": "square_hd",
    "9:16": "portrait_16_9",
    "3:4": "portrait_4_3",
    "4:3": "landscape_4_3",
    "16:9": "landscape_16_9",
    "2:3": {"width": 1024, "height": 1536},
    "3:2": {"width": 1536, "height": 1024},
    # creative_photoshoot's own size option, same constraints as above --
    # 1024x1280 is an exact 4:5 ratio, both dimensions multiples of 16.
    "4:5": {"width": 1024, "height": 1280},
}

# "color" and "target_area" are Recolor's own UI/schema fields, already
# consumed by build_instruction() (above, in the caller) to produce the
# `prompt` string -- they are not part of gpt-image-2/edit's real input
# schema (prompt, image_urls, image_size, background, quality, num_images,
# output_format, sync_mode, mask_url) and must not be forwarded raw.
# "source_feature_type" is enhance_prompt's own UI/schema field, already
# consumed by build_instruction() to look up the source tool's enhance_hint
# -- not part of openrouter/router's real input schema, must not be forwarded.
# "idea" is creative_photoshoot's own UI/schema field, already consumed by
# build_instruction() to build the scene prompt -- not part of
# gpt-image-2/edit's real input schema, must not be forwarded raw.
# "shot_type" is listing_photoshoot's own per-job planning metadata (written
# by plan_listing_shots() into each job's input_params via per_job_overrides,
# before the job ever exists -- there is no build_instruction step for this
# tool) -- not part of gpt-image-2/edit's real input schema either.
FAL_NON_SCHEMA_FIELDS = {"color", "target_area", "source_feature_type", "idea", "shot_type"}

# listing_photoshoot's own "quality" schema field (options: "standard"/
# "high") uses UI labels, not fal's real quality enum (auto/low/medium/
# high) -- "standard" is not a valid fal value at all, and while "high"
# HAPPENS to already be spelled the same as a real fal value, that's not
# guaranteed to stay true and must still be translated explicitly, not
# assumed. Kept feature-scoped, not merged into a shared quality-VALUE
# lookup (see CREATIVE_QUALITY_TIERS below) because "high" is ALSO one of
# recolor's own real, valid quality values -- a shared table would corrupt
# a real recolor "high" selection into whatever this tool maps "high" to.
#
# Found live: this used to force BOTH "standard" and "high" to "medium"
# unconditionally (FAL_HARDCODED_PARAMS) -- selecting "high" quality for
# listing_photoshoot had silently done nothing at all.
LISTING_QUALITY_TRANSLATION = {
    "standard": "medium",
    "high": "high",
}

# creative_photoshoot's own "quality" schema field is a resolution tier
# (1K/2K/4K), not one of gpt-image-2/edit's real quality values (auto/low/
# medium/high) -- unlike recolor, where the schema's quality values already
# ARE fal's real values directly. Each tier maps to BOTH the real fal
# `quality` value and its own target resolution -- 2048x2048 and 3840x2160
# both satisfy fal's image_size constraints (multiples of 16, max edge
# 3840px, ratio <= 3:1, total pixels 655,360-8,294,400).
CREATIVE_QUALITY_TIERS = {
    "1k": {"quality": "medium", "image_size": "square_hd"},
    "2k": {"quality": "high", "image_size": {"width": 2048, "height": 2048}},
    "4k": {"quality": "high", "image_size": {"width": 3840, "height": 2160}},
}


def _translate_fal_params(params: dict, feature_type: str | None = None) -> dict:
    """
    Applied only to the payload actually sent to fal.ai -- the caller's own
    dict (job.input_params) is never mutated, so the user-facing size value
    remains untouched for display and quality remains untouched for both
    display and _resolve_credit_cost's lookup (it's already fal's real value).
    """
    translated = dict(params)

    size_value = translated.pop("size", None)
    if size_value is not None:
        translated["image_size"] = FAL_SIZE_TRANSLATION.get(size_value, size_value)

    # creative_photoshoot's quality tier always sets the real fal `quality`
    # value; it only supplies image_size when no explicit aspect-ratio size
    # was chosen above (an explicit "size" selection wins for dimensions --
    # the tier's own resolution is just its sensible default).
    tier = CREATIVE_QUALITY_TIERS.get(translated.get("quality"))
    if tier:
        translated["quality"] = tier["quality"]
        if size_value is None or size_value == "original":
            translated["image_size"] = tier["image_size"]

    if feature_type == "listing_photoshoot" and "quality" in translated:
        translated["quality"] = LISTING_QUALITY_TRANSLATION.get(translated["quality"], translated["quality"])

    for field in FAL_NON_SCHEMA_FIELDS:
        translated.pop(field, None)

    return translated


async def refill_due_subscriptions(ctx):
    db = SessionLocal()
    now = datetime.now(timezone.utc)

    due = db.query(TeamSubscription).filter(
        TeamSubscription.status == "active",
        TeamSubscription.next_refill_at <= now,
    ).all()

    for team_sub in due:
        try:
            locked_sub = db.query(TeamSubscription).filter(
                TeamSubscription.id == team_sub.id
            ).populate_existing().with_for_update().first()

            if not locked_sub or locked_sub.status != "active":
                continue
            if locked_sub.next_refill_at > now:
                continue

            plan = db.query(Subscription).filter(
                Subscription.id == locked_sub.subscription_id
            ).first()
            if not plan:
                continue

            step = {"week": timedelta(weeks=1), "month": timedelta(days=30), "year": timedelta(days=30)}
            if plan.period_label not in step:
                continue

            refill_subscription_credits(db, locked_sub.team_id, locked_sub.credits_per_refill, commit=False)
            locked_sub.next_refill_at = now + step[plan.period_label]
            db.commit()
            logger.info(
                "Refilled subscription for team %s (+%s credits)",
                locked_sub.team_id, locked_sub.credits_per_refill,
            )
        except Exception:
            db.rollback()
            logger.exception(
                "refill failed for team %s — skipping, will retry next run",
                team_sub.team_id,
            )

    lapsed = (
        db.query(TeamSubscription)
        .join(Team, Team.id == TeamSubscription.team_id)
        .filter(
            TeamSubscription.status == "cancelled",
            TeamSubscription.next_refill_at <= now,
            Team.subscription_credits_remaining > 0,
        )
        .all()
    )
    for team_sub in lapsed:
        try:
            team = db.query(Team).filter(Team.id == team_sub.team_id).with_for_update().first()
            if team:
                team.subscription_credits_remaining = 0
            db.commit()
            logger.info("Lapsed cancelled subscription credits for team %s", team_sub.team_id)
        except Exception:
            db.rollback()
            logger.exception("failed to lapse cancelled subscription for team %s", team_sub.team_id)

    db.close()


async def cleanup_stale_pending_subscriptions(ctx):
    db = SessionLocal()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)

    stale = db.query(TeamSubscription).filter(
        TeamSubscription.status == "pending",
        TeamSubscription.created_at <= cutoff,
    ).all()

    for row in stale:
        logger.info("Removing stale pending subscription for team %s", row.team_id)
        db.delete(row)

    db.commit()
    db.close()


GENERIC_START_FAILED_MESSAGE = "Failed to start generation. Please try again."

# At-capacity re-enqueues use pool.enqueue_job(), NOT arq's own Retry
# mechanism -- each call starts a brand-new arq job with its own try counter
# reset to 1, so arq's max_tries does not bound this loop at all. Without an
# explicit cap here, a permanently-stuck concurrency counter (or any other
# reason try_reserve_fal_slot keeps returning False -- now only the per-team
# cap; see generation_lock.try_reserve_fal_slot) would retry every 5s
# forever. 12 attempts * 5s defer =~ 1 minute before giving up cleanly.
MAX_CAPACITY_RETRIES = 12


async def submit_generation_to_fal(ctx, job_id: str, attempt: int = 1):
    """
    Picks up a queued GenerationJob and submits it to fal.ai for real.
    On success: status -> processing, waits for the real webhook.
    On failure: status -> failed, credits refunded, lock (+ fal slot, if one
    was actually reserved) released.
    """
    db = SessionLocal()
    try:
        job = db.query(GenerationJob).filter(GenerationJob.id == job_id).with_for_update().first()
        if not job or job.status != "queued":
            return

        tool = get_tool_definition(db, job.feature_type)
        if not tool:
            return

        def fail_cleanly(message: str) -> None:
            job.status = "failed"
            job.error_message = message
            refund_credits(
                db, job.team_id, job.credits_charged,
                job.credits_from_subscription, job.credits_from_topup,
            )
            db.commit()
            release_generation_lock(job.user_id)

        # Everything from here through the fal submission must have a clean
        # failure path -- an uncaught exception anywhere in this function
        # leaves the job stuck "queued" forever with credits spent and the
        # lock held, since nothing else will ever mark it failed (the 10-min
        # sweep eventually would, but there's no reason to make a user wait
        # that long for what's already a definitive failure).
        try:
            slot_reserved = try_reserve_fal_slot(job.team_id)
        except Exception:
            logger.exception("try_reserve_fal_slot raised for job %s", job.id)
            fail_cleanly(GENERIC_START_FAILED_MESSAGE)
            return

        if not slot_reserved:
            if attempt >= MAX_CAPACITY_RETRIES:
                logger.warning(
                    "job %s gave up waiting for a fal concurrency slot after %d attempts",
                    job.id, attempt,
                )
                fail_cleanly(GENERIC_START_FAILED_MESSAGE)
                return

            # At the per-team cap right now -- re-enqueue this same job to try
            # again shortly, so one team's batch can't starve the others. (The
            # account-wide fal limit is no longer pre-checked or retried on
            # here -- fal's own documented behavior already queues and
            # dispatches automatically once that's hit, so submission below
            # just proceeds and lets fal handle it.)
            try:
                pool = await get_arq_pool()
                await pool.enqueue_job("submit_generation_to_fal", job_id, attempt + 1, _defer_by=5)
            except Exception:
                logger.exception("failed to re-enqueue job %s for a later capacity retry", job.id)
                fail_cleanly(GENERIC_START_FAILED_MESSAGE)
            return

        webhook_url = f"{settings.public_backend_url}/webhooks/fal?job_id={job.id}"

        try:
            build_instruction = TOOL_HANDLERS.get(job.feature_type)
            if build_instruction:
                # build_instruction can itself make a blocking, synchronous
                # HTTP call (recolor's vision step -> call_fal_sync). Run it
                # in a thread, not inline -- called directly, it would freeze
                # this WHOLE worker's event loop (every other job, arq's own
                # heartbeat, graceful shutdown) for however long that call
                # takes. Found live: this is very likely why a job caught
                # mid-vision-call by a worker restart was left stuck
                # "queued" instead of cleanly failed -- the loop wasn't free
                # to run the except block/cleanup below, let alone arq's own
                # shutdown handler.
                instruction = await asyncio.to_thread(build_instruction, job, tool, db)
                params = {**job.input_params, "prompt": instruction}
                # The tool's own ai_steps can name which underlying model the
                # fal endpoint itself should route to (e.g. enhance_prompt's
                # tool_definitions row: ai_steps={"model": "google/gemini-2.5-flash"}
                # for openrouter/router) -- distinct from ai_steps sub-keys
                # like recolor's detect_target, which build_instruction()
                # already fully consumes itself and never surfaces here.
                #
                # Deliberately scoped to enhance_prompt only -- unlike
                # openrouter/router, gpt-image-2/edit (recolor's and
                # creative_photoshoot's real fal model) has NO "model" field
                # in its schema at all. tool.ai_steps is a raw JSONB column
                # hand-edited via SQL with no schema enforcement; a top-level
                # "model" key added to recolor's or creative_photoshoot's row
                # by mistake (easy to do since nested steps like
                # detect_target/scene_vision already use "model" as a
                # sub-key) would otherwise silently inject an invalid field
                # into every real image generation call for that tool.
                if job.feature_type == "enhance_prompt" and tool.ai_steps.get("model"):
                    params["model"] = tool.ai_steps["model"]
            else:
                params = job.input_params

            params = _translate_fal_params(params, job.feature_type)

            # Same reasoning -- submit_to_fal() is also a blocking httpx call.
            fal_request_id = await asyncio.to_thread(submit_to_fal, tool.fal_model_id, params, webhook_url)
            job.fal_request_id = fal_request_id
            job.status = "processing"
            db.commit()
            logger.info("Submitted job %s to fal.ai (request_id=%s)", job.id, fal_request_id)
            # NOTE: the fal slot stays reserved -- it is only released once
            # this job reaches a real terminal state (in handle_fal_webhook
            # on success/failure, or below in sweep_stale_generation_jobs
            # on timeout). It is NOT released here on a successful submit.
        except Exception as e:
            fail_cleanly(f"fal submit failed: {str(e).replace(settings.fal_key, '[REDACTED]')}")
            release_fal_slot(job.team_id)
            logger.info("fal submit failed for job %s, refunded and released lock+slot", job.id)
    finally:
        db.close()


def _fail_stale_job(db, job_id, message: str) -> bool:
    """
    Shared cleanup for a single stuck job, used by both the frequent
    per-tool timeout check and the 10-minute catch-all sweep below. Re-locks
    and re-reads the row under with_for_update() + populate_existing() (NOT
    reusing the caller's unlocked copy) so a job a concurrent webhook already
    resolved a moment ago is correctly left alone instead of being
    double-refunded -- see test_sweep_populate_existing_prevents_stale_identity_map_read
    for why populate_existing() specifically is required here.

    Returns True if the job was actually failed, False if it was skipped
    (already resolved, or gone).
    """
    locked_job = db.query(GenerationJob).filter(
        GenerationJob.id == job_id
    ).populate_existing().with_for_update().first()

    if not locked_job or locked_job.status not in ("queued", "processing"):
        return False

    # A fal slot is only ever reserved once a job reaches "processing"
    # (submit_generation_to_fal). A job still "queued" here (e.g. the worker
    # never picked it up, or bailed early on an unknown tool) never held one
    # -- releasing it anyway would decrement a counter nothing incremented,
    # corrupting the concurrency cap.
    had_fal_slot = locked_job.status == "processing"

    locked_job.status = "failed"
    locked_job.error_message = message
    refund_credits(
        db, locked_job.team_id, locked_job.credits_charged,
        locked_job.credits_from_subscription, locked_job.credits_from_topup,
    )
    locked_job.completed_at = datetime.now(timezone.utc)
    db.commit()
    release_generation_lock(locked_job.user_id)
    if had_fal_slot:
        release_fal_slot(locked_job.team_id)
    return True


async def check_generation_timeouts(ctx):
    """
    Two jobs in one pass over every "processing" job, both driven by fal's own
    real queue status:

    1. DELIVERY: as soon as fal reports COMPLETED, the result is fetched and
       applied immediately -- regardless of how much of the tool's budget is
       left. This is what actually resolves jobs whenever the fal webhook
       never arrives, which is ALWAYS the case in local dev: fal's cloud
       cannot reach a PUBLIC_BACKEND_URL of 127.0.0.1, so the webhook is
       never delivered and this poll is the only path that finishes a job.
       This check used to sit behind the budget gate below, which meant a job
       fal had finished in 40s still sat "processing" until its full budget
       elapsed (60s for recolor -- looked fine; 180s for creative_photoshoot
       -- looked broken). In production it's also a genuine safety net for a
       webhook that's lost or arrives late.

    2. TIMEOUT: only once a job is past its own tool's generation_timeout_seconds
       AND fal has not reported it COMPLETED is it failed + refunded. Checking
       fal first matters: our budget elapsing doesn't mean fal's work did too,
       and blindly failing+refunding a job fal already delivered is a real
       money-loss bug.

    Only targets "processing" jobs -- a job still "queued" hasn't been
    submitted to fal yet, so there's nothing to poll and no tool-specific
    budget to measure it against; the 10-minute sweep's flat cutoff still
    catches a job stuck in that earlier state.
    """
    db = SessionLocal()
    now = datetime.now(timezone.utc)

    processing = db.query(GenerationJob).filter(
        GenerationJob.status == "processing",
    ).all()

    tools_by_feature_type = {}

    for job in processing:
        try:
            if job.feature_type not in tools_by_feature_type:
                tools_by_feature_type[job.feature_type] = get_tool_definition(db, job.feature_type)

            tool = tools_by_feature_type[job.feature_type]
            if not tool:
                continue  # unknown tool -- the 10-minute sweep will still catch it

            past_budget = job.created_at <= now - timedelta(seconds=tool.generation_timeout_seconds)

            # check_fal_status / fetch_fal_result / handle_fal_webhook (which
            # itself downloads + uploads the output) are all blocking,
            # synchronous HTTP calls -- run them in a thread so a slow fal
            # status check or storage round-trip doesn't freeze this whole
            # worker's event loop for every other job in flight.
            fal_status = None
            if job.fal_request_id:
                try:
                    status_payload = await asyncio.to_thread(
                        check_fal_status, tool.fal_model_id, job.fal_request_id,
                    )
                    fal_status = status_payload.get("status")
                except Exception:
                    logger.exception(
                        "fal status check failed for job %s (fal_request_id=%s) -- "
                        "falling back to the local timeout",
                        job.id, job.fal_request_id,
                    )

            if fal_status == "COMPLETED":
                # fal has finished (successfully or not) -- deliver the real
                # result now, through the exact same path a real webhook
                # delivery would use. Deliberately NOT gated on the budget:
                # this is what finishes jobs at all whenever the webhook never
                # arrives (always, in local dev).
                result_payload = await asyncio.to_thread(
                    fetch_fal_result, tool.fal_model_id, job.fal_request_id,
                )
                logger.info(
                    "job %s: fal reports COMPLETED -- delivering the real result "
                    "(past_budget=%s)",
                    job.id, past_budget,
                )
                await asyncio.to_thread(generation_svc.handle_fal_webhook, db, job.id, result_payload)
                continue

            if not past_budget:
                continue  # still within its own tool's budget -- let it keep running

            if fal_status in ("IN_QUEUE", "IN_PROGRESS"):
                # Measuring this specifically: how often our timeout fires
                # while fal itself confirms the work is still genuinely
                # ongoing (as opposed to us just never hearing back).
                logger.warning(
                    "job %s timed out while fal confirmed still in-progress "
                    "(fal_status=%s) -- failing per the timeout as designed",
                    job.id, fal_status,
                )

            if _fail_stale_job(db, job.id, TIMEOUT_MESSAGE):
                logger.info(
                    "Timed out job %s (feature_type=%s, budget=%ds)",
                    job.id, job.feature_type, tool.generation_timeout_seconds,
                )
        except Exception:
            db.rollback()
            logger.exception("failed to check timeout for job %s — will retry next run", job.id)

    db.close()


async def sweep_stale_generation_jobs(ctx):
    """
    Catches jobs stuck in queued/processing for too long — a crashed worker,
    a lost webhook, or fal.ai never responding. Marks them failed and refunds.
    A catch-all safety net behind check_generation_timeouts above: it uses one
    fixed 10-minute cutoff (vs. each tool's own, much shorter budget), so it
    also still catches jobs stuck "queued" (never even submitted) which the
    per-tool check above doesn't look at.
    """
    db = SessionLocal()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)

    stale = db.query(GenerationJob).filter(
        GenerationJob.status.in_(["queued", "processing"]),
        GenerationJob.created_at <= cutoff,
    ).all()

    for job in stale:
        try:
            if _fail_stale_job(db, job.id, SWEEP_TIMEOUT_MESSAGE):
                logger.info("Swept stale job %s (was stuck since %s)", job.id, job.created_at)
        except Exception:
            db.rollback()
            logger.exception("failed to sweep job %s — will retry next run", job.id)

    db.close()


async def reconcile_fal_slots(ctx):
    """
    Self-healing for the fal in-flight Redis counters (try_reserve_fal_slot /
    release_fal_slot in generation_lock.py): they're plain INCR/DECR
    counters with no TTL, unlike the generation lock. Any time the worker is
    killed between reserving and releasing a slot -- a Ctrl+C, a crash, a
    redeploy -- the leaked count sticks around forever with nothing to ever
    correct it. Confirmed live more than once this session (a per-team
    counter sitting at 1 with zero jobs actually processing).

    Recomputes the TRUE per-team in-flight count directly from Postgres (the
    real source of truth: jobs genuinely still "processing" right now) and
    resets Redis to match -- both the per-team keys and the global one.
    """
    db = SessionLocal()
    try:
        true_counts = {
            str(team_id): count
            for team_id, count in (
                db.query(GenerationJob.team_id, func.count(GenerationJob.id))
                .filter(GenerationJob.status == "processing")
                .group_by(GenerationJob.team_id)
                .all()
            )
        }

        # KEYS is O(N) over the keyspace -- fine for this app's scale run
        # every 10 minutes, but would need SCAN instead at real production
        # volume.
        remaining_team_keys = {
            key.split(":", 2)[2] for key in redis_client.keys("fal:inflight_count:*")
        }

        for team_id_str, true_count in true_counts.items():
            key = f"fal:inflight_count:{team_id_str}"
            current = int(redis_client.get(key) or 0)
            if current != true_count:
                logger.warning(
                    "reconcile_fal_slots: team %s Redis count was %d, Postgres truth "
                    "is %d -- correcting",
                    team_id_str, current, true_count,
                )
                redis_client.set(key, true_count)
            remaining_team_keys.discard(team_id_str)

        # Any team key left here has a nonzero-or-stale Redis count but zero
        # real "processing" jobs for it right now -- a pure leak, zero it.
        for team_id_str in remaining_team_keys:
            key = f"fal:inflight_count:{team_id_str}"
            current = int(redis_client.get(key) or 0)
            if current != 0:
                logger.warning(
                    "reconcile_fal_slots: team %s Redis count was %d with zero jobs "
                    "actually processing -- leaked, resetting to 0",
                    team_id_str, current,
                )
                redis_client.set(key, 0)

        corrected_global = sum(true_counts.values())
        current_global = int(redis_client.get(INFLIGHT_KEY_GLOBAL) or 0)
        if current_global != corrected_global:
            logger.warning(
                "reconcile_fal_slots: global count was %d, Postgres truth is %d -- correcting",
                current_global, corrected_global,
            )
            redis_client.set(INFLIGHT_KEY_GLOBAL, corrected_global)
    finally:
        db.close()


async def startup(ctx):
    logger.info("arq worker started")


async def shutdown(ctx):
    logger.info("arq worker shutting down")


class WorkerSettings:
    functions = [
        refill_due_subscriptions,
        cleanup_stale_pending_subscriptions,
        submit_generation_to_fal,
        sweep_stale_generation_jobs,
        check_generation_timeouts,
        reconcile_fal_slots,
    ]
    cron_jobs = [
        cron(refill_due_subscriptions, hour=3, minute=0),
        cron(cleanup_stale_pending_subscriptions, hour=4, minute=0),
        cron(sweep_stale_generation_jobs, minute={0, 15, 30, 45}),
        cron(check_generation_timeouts, second={0, 15, 30, 45}),
        cron(reconcile_fal_slots, minute={0, 10, 20, 30, 40, 50}),
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)


WorkerSettings.on_startup = startup
WorkerSettings.on_shutdown = shutdown