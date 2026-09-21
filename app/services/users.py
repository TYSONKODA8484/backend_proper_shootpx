import logging
import uuid as uuid_module
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.cache import get_cached, set_cached, clear_cached
from app.models.user import User
from app.models.team import Team
from app.models.team_member import TeamMember
from app.services.signup_abuse import (
    is_disposable_email, should_grant_signup_bonus, record_signup_bonus_grant,
    log_signup_bonus_attempt,
)

logger = logging.getLogger(__name__)

FREE_SIGNUP_CREDITS = 5  # snapshot at team-creation time — changing this later never affects existing teams

USER_CACHE_KEY_PREFIX = "user:uid:"
# Short TTL rather than a precise invalidation web: the only fields cached
# here are written exactly once, at account creation (get_or_create_user
# never updates an existing row -- a returning user's name/avatar from their
# Firebase token are deliberately not re-synced), so staleness has nothing
# to go stale against today. The TTL is the safety net for if that ever
# changes; clear_cached_user() is the explicit escape hatch.
USER_CACHE_TTL_SECONDS = 300


@dataclass(frozen=True)
class CachedUser:
    """Attribute-compatible stand-in for the User ORM row -- same trick as
    services/tool_definitions.py's CachedToolDefinition. Routes only ever
    read `user.id` / `user.email` off this (verified by grep across
    app/routes), and /auth/me serialises it through a Pydantic model with
    from_attributes=True, so a plain object is interchangeable with the real
    row. Read-only: nothing writes back to the object get_current_user
    returns."""
    id: uuid_module.UUID
    firebase_uid: str
    email: str | None
    name: str | None
    avatar_url: str | None


def _cache_user(user: User) -> None:
    set_cached(
        USER_CACHE_KEY_PREFIX + user.firebase_uid,
        {
            "id": str(user.id),
            "firebase_uid": user.firebase_uid,
            "email": user.email,
            "name": user.name,
            "avatar_url": user.avatar_url,
        },
        ttl=USER_CACHE_TTL_SECONDS,
    )


def clear_cached_user(firebase_uid: str) -> None:
    clear_cached(USER_CACHE_KEY_PREFIX + firebase_uid)


def _create_user_with_team(
    db: Session,
    firebase_uid: str,
    email: str | None,
    name: str | None,
    avatar_url: str | None,
    ip: str | None,
    device_id: str | None,
    background_tasks,
) -> User:
    # ip is only ever None for a caller that skipped the real request path
    # entirely (tests, scripts) -- treat that as "always under the cap" so
    # local/test signups still get the bonus rather than being silently
    # capped by an empty-string IP hash collision.
    bonus_eligible = (
        ip is not None
        and not is_disposable_email(email)
        and should_grant_signup_bonus(ip, device_id)
    )

    user = User(firebase_uid=firebase_uid, email=email, name=name, avatar_url=avatar_url)
    db.add(user)
    db.flush()

    team = Team(
        name=f"{name or email or 'New'}'s Team",
        topup_credits_balance=FREE_SIGNUP_CREDITS if bonus_eligible else 0,
    )
    db.add(team)
    db.flush()

    db.add(TeamMember(team_id=team.id, user_id=user.id, role="owner"))
    db.commit()

    if ip is not None:
        if bonus_eligible:
            record_signup_bonus_grant(ip, device_id)
        if background_tasks is not None:
            background_tasks.add_task(
                log_signup_bonus_attempt, db, user.id, ip, device_id, bonus_eligible,
            )

    _cache_user(user)
    return user


def get_or_create_user(
    db: Session,
    decoded_token: dict,
    ip: str | None = None,
    device_id: str | None = None,
    background_tasks=None,
) -> User:
    """Return the User for a verified Firebase token, creating it (plus a personal
    team) on first login. Safe against concurrent first requests.

    ip/device_id/background_tasks are only used the ONE time this actually
    creates an account (the anti-abuse signup-bonus cap) -- every subsequent
    call for an already-existing user ignores them entirely.
    """
    firebase_uid = decoded_token["uid"]

    # This lookup runs on EVERY authenticated request, and at the measured
    # ~160ms round trip to the (Tokyo-hosted) database it was the single
    # largest fixed cost in the app -- paid before any endpoint's own work
    # even started. Redis answers it in ~1ms.
    cached = get_cached(USER_CACHE_KEY_PREFIX + firebase_uid)
    if cached is not None:
        return CachedUser(
            id=uuid_module.UUID(cached["id"]),
            firebase_uid=cached["firebase_uid"],
            email=cached["email"],
            name=cached["name"],
            avatar_url=cached["avatar_url"],
        )

    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    if user:
        _cache_user(user)
        return user

    try:
        return _create_user_with_team(
            db,
            firebase_uid=firebase_uid,
            email=decoded_token.get("email"),
            name=decoded_token.get("name"),
            avatar_url=decoded_token.get("picture"),
            ip=ip,
            device_id=device_id,
            background_tasks=background_tasks,
        )
    except IntegrityError:
        db.rollback()
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        if user:
            _cache_user(user)
            return user
        logger.exception("user provisioning failed for %s", firebase_uid)
        raise