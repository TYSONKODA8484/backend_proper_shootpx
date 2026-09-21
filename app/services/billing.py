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


def is_unpaid_checkout(team_sub) -> bool:
    """
    True for a subscription row that exists only because a checkout was STARTED
    and never paid for: status "pending" with credits_per_refill still 0
    (activation is the only thing that sets it above 0 -- see
    webhooks.handle_subscription_activated).

    "pending" is deliberately not enough on its own, because it also means
    something completely different: handle_subscription_pending marks a
    genuinely ACTIVE, paying subscription "pending" when a renewal payment
    fails and Razorpay retries for days. Treating that customer's row as an
    abandoned checkout would let it be replaced or, in the old cleanup cron,
    deleted outright.
    """
    return team_sub is not None and team_sub.status == "pending" and (team_sub.credits_per_refill or 0) == 0


def _cancel_abandoned_razorpay_subscription(razorpay_subscription_id: str) -> None:
    """Best-effort. A never-paid ("created") Razorpay subscription can never
    charge anyone, so failing to cancel it must not block the user's retry --
    it is logged and left to expire."""
    try:
        razorpay_client.subscription.cancel(razorpay_subscription_id)
    except Exception:
        logger.warning(
            "could not cancel abandoned Razorpay subscription %s -- harmless (never paid), left as is",
            razorpay_subscription_id, exc_info=True,
        )




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
    replaced_razorpay_id = None
    if row is not None and row.status in ("active", "pending"):
        if not is_unpaid_checkout(row):
            # a real subscription: active, or paying but mid renewal-retry
            raise ValueError("This team already has an active subscription")

        # An abandoned, never-paid checkout (the user closed Razorpay). This
        # must NOT block them -- it was never a subscription.
        if row.subscription_id == subscription_id and row.razorpay_subscription_id:
            # Same plan again ("Retry payment", or a double click): hand back
            # the SAME Razorpay subscription if it is still payable, instead of
            # creating a second one and cancelling the first out from under a
            # checkout window the user may have open.
            try:
                existing = razorpay_client.subscription.fetch(row.razorpay_subscription_id)
            except Exception:
                existing = None
            if existing and existing.get("status") == "created":
                return {
                    "razorpay_subscription_id": row.razorpay_subscription_id,
                    "key_id": settings.razorpay_key_id,
                }
        # Different plan, or the old one is no longer payable: replace it.
        replaced_razorpay_id = row.razorpay_subscription_id

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
    # A fresh lifecycle needs its own renewal notice -- without resetting
    # this, a team resubscribing after a yearly plan completed (see
    # worker.py's send_yearly_renewal_notices) would carry over the OLD
    # cycle's sent_at and never get warned before the new cycle also ends.
    row.renewal_notice_sent_at = None

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

    # Only AFTER the replacement is committed, so a failure creating it leaves
    # the user's previous (still payable) attempt untouched.
    if replaced_razorpay_id:
        _cancel_abandoned_razorpay_subscription(replaced_razorpay_id)

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

    if is_unpaid_checkout(team_sub):
        # Never paid, so there is no billing to stop and nothing that can go
        # wrong by cancelling locally. Razorpay is told on a best-effort basis
        # only: an error there must not leave the user stuck with an "already
        # in progress" attempt they are trying to get rid of.
        if team_sub.razorpay_subscription_id:
            _cancel_abandoned_razorpay_subscription(team_sub.razorpay_subscription_id)
        team_sub.status = "cancelled"
        db.commit()
        return team_sub

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

    existing = db.query(TeamSubscription).filter(TeamSubscription.team_id == team_id).first()
    if is_unpaid_checkout(existing):
        # "Switching" away from a checkout that was never paid is just starting
        # a fresh checkout: nothing to cancel, and no leftover credits to carry
        # over (a never-activated subscription never granted any).
        return create_subscription_checkout(db, team_id, new_subscription_id)

    cancel_subscription(db, team_id)

    team = db.query(Team).filter(Team.id == team_id).with_for_update().first()
    if not team:
        raise ValueError("Team not found")

    leftover = team.subscription_credits_remaining
    team.topup_credits_balance += leftover
    team.subscription_credits_remaining = 0
    db.commit()

    return create_subscription_checkout(db, team_id, new_subscription_id)