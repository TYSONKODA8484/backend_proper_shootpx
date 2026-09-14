import logging
logging.basicConfig(level=logging.INFO)

from datetime import datetime, timedelta, timezone

from arq.connections import RedisSettings
from arq.cron import cron

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.fal_client import submit_to_fal, check_fal_status, fetch_fal_result
from app.core.arq_pool import get_arq_pool
from app.models.team_subscription import TeamSubscription
from app.models.subscription import Subscription
from app.models.team import Team
from app.models.generation_job import GenerationJob
from app.models.tool_definition import ToolDefinition
from app.services.credits import refill_subscription_credits, refund_credits
from app.services.generation_lock import release_generation_lock, try_reserve_fal_slot, release_fal_slot
from app.services import generation as generation_svc
from app.services.generation import TIMEOUT_MESSAGE, SWEEP_TIMEOUT_MESSAGE
from app.tools.registry import TOOL_HANDLERS

logger = logging.getLogger(__name__)

# Our param_schema exposes user-facing values (Recolor's Photoroom-style
# standard/advanced/premium tiers, plain aspect-ratio strings) that must stay
# exactly as-is -- they're what's stored in job.input_params for display and
# what _resolve_credit_cost keys off of. fal.ai's real API takes neither
# verbatim: openai/gpt-image-2/edit's `quality` only accepts
# auto/low/medium/high, and its `image_size` only accepts a fixed set of
# named presets (or an explicit {width, height} object) -- confirmed against
# https://fal.ai/models/openai/gpt-image-2/edit/api. These tables are
# currently written against exactly that model's vocabulary; if a future tool
# reuses the "quality"/"size" field names with different values against a
# different fal model, this translation would need to become per-tool rather
# than global.
FAL_QUALITY_TRANSLATION = {
    "standard": "low",
    "advanced": "medium",
    "premium": "high",
}

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
}

# "color" and "target_area" are Recolor's own UI/schema fields, already
# consumed by build_instruction() (above, in the caller) to produce the
# `prompt` string -- they are not part of gpt-image-2/edit's real input
# schema (prompt, image_urls, image_size, background, quality, num_images,
# output_format, sync_mode, mask_url) and must not be forwarded raw.
FAL_NON_SCHEMA_FIELDS = {"color", "target_area"}


def _translate_fal_params(params: dict) -> dict:
    """
    Applied only to the payload actually sent to fal.ai -- the caller's own
    dict (job.input_params) is never mutated, so the user-facing quality/size
    values remain untouched for display and for credit-cost lookup.
    """
    translated = dict(params)

    if "quality" in translated:
        translated["quality"] = FAL_QUALITY_TRANSLATION.get(translated["quality"], translated["quality"])

    if "size" in translated:
        size = translated.pop("size")
        translated["image_size"] = FAL_SIZE_TRANSLATION.get(size, size)

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

        tool = db.query(ToolDefinition).filter(
            ToolDefinition.feature_type == job.feature_type
        ).first()
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
                instruction = build_instruction(job, tool)
                params = {**job.input_params, "prompt": instruction}
            else:
                params = job.input_params

            params = _translate_fal_params(params)

            fal_request_id = submit_to_fal(tool.fal_model_id, params, webhook_url)
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
    Per-tool timeout enforcement: runs far more often than the 10-minute
    catch-all sweep below, and times each "processing" job out against its
    own tool's generation_timeout_seconds instead of one fixed cutoff for
    every tool. Only targets "processing" jobs -- a job still "queued" hasn't
    been submitted to fal yet, so there's no tool-specific budget to measure
    it against; the 10-minute sweep's flat cutoff still catches a job stuck
    in that earlier state.

    Before actually failing a job for running past its budget, this confirms
    against fal's own real queue status first -- our local timeout elapsing
    doesn't mean fal's work did too, and blindly failing+refunding a job fal
    is about to (or already did) deliver for free is a real money-loss bug.
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
                tools_by_feature_type[job.feature_type] = db.query(ToolDefinition).filter(
                    ToolDefinition.feature_type == job.feature_type
                ).first()

            tool = tools_by_feature_type[job.feature_type]
            if not tool:
                continue  # unknown tool -- the 10-minute sweep will still catch it

            if job.created_at > now - timedelta(seconds=tool.generation_timeout_seconds):
                continue  # still within its own tool's budget

            fal_status = None
            if job.fal_request_id:
                try:
                    status_payload = check_fal_status(tool.fal_model_id, job.fal_request_id)
                    fal_status = status_payload.get("status")
                except Exception:
                    logger.exception(
                        "fal status check failed for job %s (fal_request_id=%s) -- "
                        "falling back to the local timeout",
                        job.id, job.fal_request_id,
                    )

            if fal_status == "COMPLETED":
                # fal actually finished (successfully or not) before we got to
                # it -- resolve with the real result via the exact same path a
                # real webhook delivery would use, instead of failing a job
                # that may well have already succeeded.
                result_payload = fetch_fal_result(tool.fal_model_id, job.fal_request_id)
                logger.info(
                    "job %s locally timed out but fal had already COMPLETED it -- "
                    "resolving with the real result instead of failing",
                    job.id,
                )
                generation_svc.handle_fal_webhook(db, job.id, result_payload)
                continue

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
    ]
    cron_jobs = [
        cron(refill_due_subscriptions, hour=3, minute=0),
        cron(cleanup_stale_pending_subscriptions, hour=4, minute=0),
        cron(sweep_stale_generation_jobs, minute={0, 15, 30, 45}),
        cron(check_generation_timeouts, second={0, 15, 30, 45}),
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)


WorkerSettings.on_startup = startup
WorkerSettings.on_shutdown = shutdown