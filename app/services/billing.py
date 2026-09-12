import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.razorpay_client import razorpay_client
from app.models.credit import Credit
from app.models.subscription import Subscription
from app.models.team_subscription import TeamSubscription
from app.core.config import settings
from app.models.team import Team

logger = logging.getLogger(__name__)


class RazorpayCancelError(RuntimeError):
    """Razorpay rejected or failed the subscription-cancel call. Distinct from
    ValueError so callers can tell 'nothing to cancel' apart from 'the provider
    call failed' (delete_team logs-and-continues on the latter)."""




def create_credit_pack_checkout(db: Session, team_id, pack_id) -> dict:
    # The charge amount and the credit count are ALWAYS taken from the catalog row
    # here — never from anything the caller passes in. The checkout route accepts
    # no request body at all; `pack_id` is the only client input and it is just a
    # lookup key.
    pack = db.query(Credit).filter(Credit.id == pack_id).first()
    if not pack:
        raise ValueError("Credit pack not found")

    order = razorpay_client.order.create({
        "amount": pack.price,       # already in paise, from the DB
        "currency": "INR",
        "notes": {
            # Razorpay note values must be strings. `team_id` / `credit_pack_id`
            # are what the webhook uses to identify the purchase; `credits` is
            # informational only (the webhook re-reads it from the DB, never here).
            "team_id": str(team_id),
            "credit_pack_id": str(pack_id),
            "credits": str(pack.credits),
        },
    })

    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "key_id": settings.razorpay_key_id,
    }

def create_subscription_checkout(db: Session, team_id, subscription_id) -> dict:
    plan = db.query(Subscription).filter(Subscription.id == subscription_id).first()
    if not plan:
        raise ValueError("Plan not found")
    if not plan.razorpay_plan_id:
        raise ValueError("This plan is not configured for payment yet")

    try:
        total_count = {"week": 52, "month": 12, "year": 1}[plan.period_label]
    except KeyError:
        raise ValueError(f"Unsupported billing period: {plan.period_label!r}")

    now = datetime.now(timezone.utc)

    # Lock this team's subscription row (if it has one) for the whole checkout so
    # two rapid "Subscribe" clicks are serialised: the second waits here, then
    # sees the pending/active row the first one wrote and is rejected.
    row = (
        db.query(TeamSubscription)
        .filter(TeamSubscription.team_id == team_id)
        .with_for_update()
        .first()
    )
    if row is not None and row.status in ("active", "pending"):
        raise ValueError("This team already has an active subscription")

    # Claim a `pending` row BEFORE calling Razorpay — no external subscription is
    # created until this claim is secured. A brand-new team has no row, so two
    # concurrent requests race to INSERT and UNIQUE(team_id) rejects the loser; a
    # team resubscribing after cancel/halt reuses its existing row (a fresh
    # lifecycle) and the FOR UPDATE lock above serialises the requests.
    if row is None:
        row = TeamSubscription(team_id=team_id)
        db.add(row)
    row.subscription_id = subscription_id
    row.razorpay_subscription_id = None
    row.status = "pending"
    row.credits_per_refill = 0
    row.next_refill_at = now
    row.current_period_end = now

    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise ValueError("This team already has an active subscription")

    # If Razorpay itself fails, roll the claim back so a stuck `pending` row never
    # blocks future checkout attempts.
    try:
        razor_sub = razorpay_client.subscription.create({
            "plan_id": plan.razorpay_plan_id,
            "total_count": total_count,
            "notes": {
                "team_id": str(team_id),
                "subscription_id": str(subscription_id),
            },
        })
    except Exception:
        db.rollback()
        raise

    # Store the real id NOW (not None) so a cancel during the pending window can
    # actually reach Razorpay, and so a cancelled row can't be resurrected by a
    # late subscription.activated webhook.
    row.razorpay_subscription_id = razor_sub["id"]
    db.commit()

    return {
        "razorpay_subscription_id": razor_sub["id"],
        "key_id": settings.razorpay_key_id,
    }

def cancel_subscription(db: Session, team_id) -> TeamSubscription:
    team_sub = db.query(TeamSubscription).filter(
        TeamSubscription.team_id == team_id,
        TeamSubscription.status.in_(["active", "pending"]),
    ).with_for_update().first()

    if not team_sub:
        raise ValueError("This team has no active subscription to cancel")

    # Tell Razorpay to stop billing BEFORE we touch local state. If this fails we
    # leave status untouched — never show 'cancelled' while the card is still
    # being charged.
    if team_sub.razorpay_subscription_id:
        try:
            razorpay_client.subscription.cancel(team_sub.razorpay_subscription_id)
        except Exception as e:
            logger.exception(
                "Failed to cancel Razorpay subscription %s for team %s",
                team_sub.razorpay_subscription_id, team_id,
            )
            raise RazorpayCancelError(
                "Could not cancel the subscription with Razorpay. Please try again."
            ) from e

    team_sub.status = "cancelled"
    db.commit()
    return team_sub

def switch_subscription(db: Session, team_id, new_subscription_id) -> dict:
    new_plan = db.query(Subscription).filter(Subscription.id == new_subscription_id).first()
    if not new_plan:
        raise ValueError("New plan not found")
    if not new_plan.razorpay_plan_id:
        raise ValueError("This plan is not configured for payment yet")

    cancel_subscription(db, team_id)

    team = db.query(Team).filter(Team.id == team_id).with_for_update().first()
    if not team:
        raise ValueError("Team not found")

    leftover = team.subscription_credits_remaining
    team.topup_credits_balance += leftover
    team.subscription_credits_remaining = 0
    db.commit()

    return create_subscription_checkout(db, team_id, new_subscription_id)