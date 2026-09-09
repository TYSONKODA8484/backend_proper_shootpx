from sqlalchemy.orm import Session
from app.models.team import Team


def get_total_credits(team: Team) -> int:
    return team.subscription_credits_remaining + team.topup_credits_balance


def add_topup_credits(db: Session, team_id, amount: int, commit: bool = True) -> Team:
    """Credit pack purchase — permanent, never expires.

    Pass commit=False when the caller needs the increment to be part of a larger
    atomic transaction (e.g. the webhook, which claims the payment id in the same
    transaction so a retried/concurrent webhook can't double-credit).
    """
    team = db.query(Team).filter(Team.id == team_id).with_for_update().first()
    if not team:
        raise ValueError("Team not found")

    team.topup_credits_balance += amount
    if commit:
        db.commit()
    return team


def refill_subscription_credits(db: Session, team_id, amount: int) -> Team:
    """Scheduled periodic refill — replaces, doesn't add. Unused credits lapse."""
    team = db.query(Team).filter(Team.id == team_id).with_for_update().first()
    if not team:
        raise ValueError("Team not found")

    team.subscription_credits_remaining = amount
    db.commit()
    return team


def spend_credits(db: Session, team_id, amount: int) -> Team:
    """Deduct for a generation. Drains subscription pool first, then topup wallet."""
    team = db.query(Team).filter(Team.id == team_id).with_for_update().first()
    if not team:
        raise ValueError("Team not found")

    total_available = team.subscription_credits_remaining + team.topup_credits_balance
    if total_available < amount:
        raise ValueError(
            f"Insufficient credits. Available: {total_available}, needed: {amount}"
        )

    from_subscription = min(team.subscription_credits_remaining, amount)
    from_topup = amount - from_subscription

    team.subscription_credits_remaining -= from_subscription
    team.topup_credits_balance -= from_topup

    db.commit()
    return team


def refund_credits(db: Session, team_id, amount: int, from_subscription: int, from_topup: int) -> Team:
    """
    Reverse a spend when fal.ai didn't actually charge us.
    Must reverse into the SAME pools the deduction came from, in the same split.
    """
    team = db.query(Team).filter(Team.id == team_id).with_for_update().first()
    if not team:
        raise ValueError("Team not found")

    team.subscription_credits_remaining += from_subscription
    team.topup_credits_balance += from_topup

    db.commit()
    return team