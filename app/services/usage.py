from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.cache import get_cached, set_cached
from app.models.generation_job import GenerationJob
from app.models.user import User
from app.models.tool_definition import ToolDefinition


def _resolve_range(period: str | None, from_dt, to_dt):
    """"week"/"month" are calendar-aligned (matches the wireframe's "This
    week"/"This month" dropdown, not a rolling N-day window) -- explicit
    from_dt/to_dt (the "Custom" range) always wins over a preset."""
    if from_dt or to_dt:
        return from_dt, to_dt

    now = datetime.now(timezone.utc)
    if period == "week":
        start = now - timedelta(days=now.weekday())  # this week's Monday
        return start.replace(hour=0, minute=0, second=0, microsecond=0), None
    # default: current calendar month
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0), None


_DISPLAY_NAME_CACHE_KEY = "landing:tool-display-names"


def _tool_display_names(db: Session) -> dict:
    """feature_type -> display_name, cached in Redis. Effectively static
    reference data (tool_definitions rows are edited by hand via SQL), and
    the "landing:" key prefix means POST /admin/cache/clear already flushes
    it alongside the other tool caches -- no new invalidation path to
    remember."""
    cached = get_cached(_DISPLAY_NAME_CACHE_KEY)
    if cached is not None:
        return cached

    names = {
        feature_type: display_name
        for feature_type, display_name in db.query(
            ToolDefinition.feature_type, ToolDefinition.display_name,
        ).all()
    }
    set_cached(_DISPLAY_NAME_CACHE_KEY, names, ttl=300)
    return names


def get_team_usage(
    db: Session, team_id, period: str | None = None, from_dt=None, to_dt=None,
) -> dict:
    """
    Only 'completed' jobs count -- a failed job's credits_charged is fully
    refunded (see fail_and_release/refund_credits) but the column itself is
    never reset to 0, so including failed/queued/processing jobs here would
    count credits that were never actually spent.
    """
    resolved_from, resolved_to = _resolve_range(period, from_dt, to_dt)

    base_query = db.query(GenerationJob).filter(
        GenerationJob.team_id == team_id, GenerationJob.status == "completed",
    )
    if resolved_from:
        base_query = base_query.filter(GenerationJob.created_at >= resolved_from)
    if resolved_to:
        base_query = base_query.filter(GenerationJob.created_at <= resolved_to)

    # --- by member: ONE query, User joined in ---------------------------------
    # Joining User is safe here (users.id is its PK, so the join can never
    # multiply a generation_jobs row and inflate the SUM) -- unlike
    # tool_definitions below. LEFT join so a job whose user row somehow went
    # missing still reports its credits rather than vanishing from the total.
    credits_sum = func.sum(GenerationJob.credits_charged)
    by_member_rows = (
        base_query.with_entities(GenerationJob.user_id, credits_sum, User.name, User.email)
        .outerjoin(User, User.id == GenerationJob.user_id)
        .group_by(GenerationJob.user_id, User.name, User.email)
        .order_by(credits_sum.desc())
        .all()
    )
    by_member = [
        {"userId": str(user_id), "name": name or email, "credits": int(credits)}
        for user_id, credits, name, email in by_member_rows
    ]

    # --- by tool: ONE query, display names resolved WITHOUT a join ------------
    # tool_definitions' PK is composite (feature_type, stage), so joining it
    # on feature_type alone could match multiple stage rows and multiply the
    # pre-aggregation rows -- silently inflating every credit total. The
    # display-name map is tiny and effectively static, so it's cached in
    # Redis (~1ms) instead of costing a round trip to resolve.
    by_tool_rows = (
        base_query.with_entities(GenerationJob.feature_type, credits_sum)
        .group_by(GenerationJob.feature_type)
        .order_by(credits_sum.desc())
        .all()
    )
    display_names = _tool_display_names(db)
    by_tool = [
        {
            "featureType": feature_type,
            "displayName": display_names.get(feature_type) or feature_type,
            "credits": int(credits),
        }
        for feature_type, credits in by_tool_rows
    ]

    # Derived, not a third aggregate query -- byTool is a complete partition
    # of the exact same filtered job set, so its sum IS the total by
    # definition (the existing test asserting creditsUsed == sum(byTool)
    # now holds structurally rather than by coincidence).
    total = sum(t["credits"] for t in by_tool)

    return {
        "creditsUsed": total,
        "byMember": by_member,
        "byTool": by_tool,
    }
