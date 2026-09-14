import logging
logging.basicConfig(level=logging.INFO)

from datetime import datetime, timedelta, timezone

from arq.connections import RedisSettings
from arq.cron import cron

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.fal_client import submit_to_fal
from app.core.arq_pool import get_arq_pool
from app.models.team_subscription import TeamSubscription
from app.models.subscription import Subscription
from app.models.team import Team
from app.models.generation_job import GenerationJob
from app.models.tool_definition import ToolDefinition
from app.services.credits import refill_subscription_credits, refund_credits
from app.services.generation_lock import release_generation_lock, try_reserve_fal_slot, release_fal_slot
from app.tools.registry import TOOL_HANDLERS

logger = logging.getLogger(__name__)


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


async def submit_generation_to_fal(ctx, job_id: str):
    """
    Picks up a queued GenerationJob and submits it to fal.ai for real.
    On success: status -> processing, waits for the real webhook.
    On failure: status -> failed, credits refunded, lock + fal slot released.
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

        if not try_reserve_fal_slot(job.team_id):
            # At capacity right now -- re-enqueue this same job to try again
            # shortly, rather than sending a request fal would just queue anyway.
            pool = await get_arq_pool()
            await pool.enqueue_job("submit_generation_to_fal", job_id, _defer_by=5)
            return

        webhook_url = f"{settings.public_backend_url}/webhooks/fal?job_id={job.id}"

        try:
            build_instruction = TOOL_HANDLERS.get(job.feature_type)
            if build_instruction:
                instruction = build_instruction(job, tool)
                params = {**job.input_params, "prompt": instruction}
            else:
                params = job.input_params

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
            job.status = "failed"
            job.error_message = f"fal submit failed: {str(e).replace(settings.fal_key, '[REDACTED]')}"
            refund_credits(
                db, job.team_id, job.credits_charged,
                job.credits_from_subscription, job.credits_from_topup,
            )
            db.commit()
            release_generation_lock(job.user_id)
            release_fal_slot(job.team_id)
            logger.info("fal submit failed for job %s, refunded and released lock+slot", job.id)
    finally:
        db.close()


async def sweep_stale_generation_jobs(ctx):
    """
    Catches jobs stuck in queued/processing for too long — a crashed worker,
    a lost webhook, or fal.ai never responding. Marks them failed and refunds.
    """
    db = SessionLocal()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)

    stale = db.query(GenerationJob).filter(
        GenerationJob.status.in_(["queued", "processing"]),
        GenerationJob.created_at <= cutoff,
    ).all()

    for job in stale:
        try:
            locked_job = db.query(GenerationJob).filter(
                GenerationJob.id == job.id
            ).populate_existing().with_for_update().first()

            if not locked_job or locked_job.status not in ("queued", "processing"):
                continue

            # A fal slot is only ever reserved once a job reaches "processing"
            # (submit_generation_to_fal). A job still "queued" here (e.g. the
            # worker never picked it up, or bailed early on an unknown tool)
            # never held one -- releasing it anyway would decrement a counter
            # nothing incremented, corrupting the concurrency cap.
            had_fal_slot = locked_job.status == "processing"

            locked_job.status = "failed"
            locked_job.error_message = "Generation timed out — no response received"
            refund_credits(
                db, locked_job.team_id, locked_job.credits_charged,
                locked_job.credits_from_subscription, locked_job.credits_from_topup,
            )
            locked_job.completed_at = datetime.now(timezone.utc)
            db.commit()
            release_generation_lock(locked_job.user_id)
            if had_fal_slot:
                release_fal_slot(locked_job.team_id)
            logger.info("Swept stale job %s (was stuck since %s)", locked_job.id, locked_job.created_at)
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
    ]
    cron_jobs = [
        cron(refill_due_subscriptions, hour=3, minute=0),
        cron(cleanup_stale_pending_subscriptions, hour=4, minute=0),
        cron(sweep_stale_generation_jobs, minute={0, 15, 30, 45}),
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)


WorkerSettings.on_startup = startup
WorkerSettings.on_shutdown = shutdown