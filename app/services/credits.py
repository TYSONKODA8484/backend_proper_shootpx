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


def refill_subscription_credits(db: Session, team_id, amount: int, commit: bool = True) -> Team:
    """Scheduled periodic refill — replaces, doesn't add. Unused credits lapse.

    Pass commit=False when the caller needs the balance change committed in the
    same transaction as something else (e.g. the refill worker, which advances
    next_refill_at atomically with the top-up).
    """
    team = db.query(Team).filter(Team.id == team_id).with_for_update().first()
    if not team:
        raise ValueError("Team not found")

    team.subscription_credits_remaining = amount
    if commit:
        db.commit()
    return team


def spend_credits(db: Session, team_id, amount: int, commit: bool = True):
    """Pass commit=False when the caller needs the deduction committed atomically
    with something else (e.g. generation job creation, so a failure creating the
    job can never leave credits spent with no job to account for them)."""
    team = db.query(Team).filter(Team.id == team_id).with_for_update().first()
    if not team:
        raise ValueError("Team not found")

    total_available = team.subscription_credits_remaining + team.topup_credits_balance
    if total_available < amount:
        raise ValueError(f"Insufficient credits. Available: {total_available}, needed: {amount}")

    from_subscription = min(team.subscription_credits_remaining, amount)
    from_topup = amount - from_subscription

    team.subscription_credits_remaining -= from_subscription
    team.topup_credits_balance -= from_topup

    if commit:
        db.commit()
    return team, from_subscription, from_topup


def spend_credits_up_to(db: Session, team_id, per_unit_cost: int, requested_units: int, commit: bool = True):
    """
    For a multi-unit batch (e.g. listing_photoshoot's N shots, each priced at
    per_unit_cost): spends for as many WHOLE units as the team can actually
    afford right now, up to requested_units, instead of spend_credits' plain
    all-or-nothing "spend exactly this total or raise". A single-unit request
    (requested_units=1) still behaves exactly like spend_credits -- either
    that one unit is affordable or this raises the same "Insufficient
    credits" error.

    The team row is locked for the whole check-and-spend (same as
    spend_credits), so the balance read here is the true balance at the
    moment of spending -- not a stale read from before this call started.
    That matters specifically because this is a shared TEAM balance: if
    another request against the same team spent credits concurrently between
    when the caller last checked the balance and this call, that spend is
    already reflected in what's available here, and this batch is sized down
    accordingly rather than racing it.

    Returns (team, granted_units, from_subscription, from_topup).
    Raises ValueError if not even one unit is affordable.
    """
    team = db.query(Team).filter(Team.id == team_id).with_for_update().first()
    if not team:
        raise ValueError("Team not found")

    total_available = team.subscription_credits_remaining + team.topup_credits_balance
    # A genuinely free tool (per_unit_cost <= 0, e.g. a flat
    # credit_cost_per_output of 0) grants the full request regardless of
    # balance -- `total_available // per_unit_cost` would otherwise divide by
    # zero, or (for a hypothetical negative cost) grant a nonsensical amount.
    if per_unit_cost <= 0:
        granted_units = requested_units
    else:
        granted_units = min(requested_units, total_available // per_unit_cost)

    if granted_units < 1:
        raise ValueError(f"Insufficient credits. Available: {total_available}, needed: {per_unit_cost}")

    total_cost = per_unit_cost * granted_units
    from_subscription = min(team.subscription_credits_remaining, total_cost)
    from_topup = total_cost - from_subscription

    team.subscription_credits_remaining -= from_subscription
    team.topup_credits_balance -= from_topup

    if commit:
        db.commit()
    return team, granted_units, from_subscription, from_topup


def refund_credits(db: Session, team_id, amount: int, from_subscription: int, from_topup: int, commit: bool = True) -> Team:
    """
    Reverse a spend when fal.ai didn't actually charge us.
    Must reverse into the SAME pools the deduction came from, in the same split.
    """
    team = db.query(Team).filter(Team.id == team_id).with_for_update().first()
    if not team:
        raise ValueError("Team not found")

    team.subscription_credits_remaining += from_subscription
    team.topup_credits_balance += from_topup

    if commit:
        db.commit()
    return team