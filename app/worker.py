import logging
logging.basicConfig(level=logging.INFO)
from datetime import datetime, timedelta, timezone

from arq.connections import RedisSettings
from arq.cron import cron

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.team import Team
from app.models.team_subscription import TeamSubscription
from app.models.subscription import Subscription
from app.services.credits import refill_subscription_credits

logger = logging.getLogger(__name__)


# calendar step per plan period — next_refill_at is always advanced from NOW,
# never from the old (possibly stale) value, so worker downtime can't cause a
# catch-up burst of refills.
_STEP = {
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
    "year": timedelta(days=30),   # yearly plans deliver a monthly slice
}


async def refill_due_subscriptions(ctx):
    """
    Runs daily. Two passes:
      1. active subscriptions past next_refill_at  -> refill + advance the date
         (delivers months 2-12 of a yearly plan, and covers a missed
         subscription.charged webhook)
      2. cancelled subscriptions past next_refill_at with credits still in the
         pool -> lapse those credits to zero (PRD §5 "use it or lose it, even
         after cancellation")
    """
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        due = db.query(TeamSubscription).filter(
            TeamSubscription.status == "active",
            TeamSubscription.next_refill_at <= now,
        ).all()

        for team_sub in due:
            try:
                # Re-fetch and lock THIS row fresh — team_sub here is from the batch
                # query taken at the top of this function, which may now be stale
                # (e.g. the owner cancelled while an earlier row in this loop was
                # being processed). Acting on the stale snapshot instead of this
                # fresh, locked read is exactly what would let a cancelled team get
                # refilled.
                locked_sub = db.query(TeamSubscription).filter(
                    TeamSubscription.id == team_sub.id
                ).with_for_update().first()

                if not locked_sub or locked_sub.status != "active":
                    continue  # cancelled/switched since the batch query ran
                if locked_sub.next_refill_at > now:
                    continue  # already refilled by something else in the meantime

                plan = db.query(Subscription).filter(
                    Subscription.id == locked_sub.subscription_id
                ).first()
                if not plan or plan.period_label not in _STEP:
                    continue

                refill_subscription_credits(
                    db, locked_sub.team_id, locked_sub.credits_per_refill, commit=False
                )
                locked_sub.next_refill_at = now + _STEP[plan.period_label]
                db.commit()
                logger.info(
                    "Refilled subscription for team %s (=%s credits)",
                    locked_sub.team_id, locked_sub.credits_per_refill,
                )
            except Exception:
                db.rollback()
                logger.exception(
                    "refill failed for team %s — skipping, will retry next run",
                    team_sub.team_id,
                )
    
        # Cancelled subscriptions: leftover subscription-pool credits stay usable
        # until the moment they'd have naturally reset (next_refill_at) — then
        # they lapse to zero, per PRD §5. Only the subscription pool is touched,
        # never topup_credits_balance. The `> 0` filter means a team is processed
        # once, not re-visited every night after it's been zeroed.
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
                team = db.query(Team).filter(
                    Team.id == team_sub.team_id
                ).with_for_update().first()
                if team:
                    team.subscription_credits_remaining = 0
                db.commit()
                logger.info(
                    "Lapsed cancelled subscription credits for team %s", team_sub.team_id
                )
            except Exception:
                db.rollback()
                logger.exception(
                    "failed to lapse cancelled subscription for team %s", team_sub.team_id
                )
    finally:
        db.close()


async def cleanup_stale_pending_subscriptions(ctx):
    """
    A `pending` row with no activation ever received blocks that team from
    subscribing again. Anything pending for more than 30 minutes is stale —
    the checkout clearly didn't complete — so it's safe to delete. Only
    `pending` rows are ever touched; active/cancelled/halted are untouched.
    """
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
        stale = db.query(TeamSubscription).filter(
            TeamSubscription.status == "pending",
            TeamSubscription.created_at <= cutoff,
        ).all()

        for row in stale:
            logger.info("Removing stale pending subscription for team %s", row.team_id)
            db.delete(row)

        db.commit()
    finally:
        db.close()


async def startup(ctx):
    logger.info("arq worker started")


async def shutdown(ctx):
    logger.info("arq worker shutting down")


class WorkerSettings:
    functions = [refill_due_subscriptions, cleanup_stale_pending_subscriptions]
    cron_jobs = [
        cron(refill_due_subscriptions, hour=3, minute=0),   # once daily, 3am
        cron(cleanup_stale_pending_subscriptions, hour=4, minute=0),
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)


WorkerSettings.on_startup = startup
WorkerSettings.on_shutdown = shutdown