import logging

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.models.user import User
from app.models.team import Team
from app.models.team_member import TeamMember

logger = logging.getLogger(__name__)

FREE_SIGNUP_CREDITS = 5  # snapshot at team-creation time — changing this later never affects existing teams


def _create_user_with_team(
    db: Session,
    firebase_uid: str,
    email: str | None,
    name: str | None,
    avatar_url: str | None,
) -> User:
    user = User(firebase_uid=firebase_uid, email=email, name=name, avatar_url=avatar_url)
    db.add(user)
    db.flush()

    team = Team(
        name=f"{name or email or 'New'}'s Team",
        topup_credits_balance=FREE_SIGNUP_CREDITS,
    )
    db.add(team)
    db.flush()

    db.add(TeamMember(team_id=team.id, user_id=user.id, role="owner"))
    db.commit()
    return user


def get_or_create_user(db: Session, decoded_token: dict) -> User:
    """Return the User for a verified Firebase token, creating it (plus a personal
    team) on first login. Safe against concurrent first requests."""
    firebase_uid = decoded_token["uid"]

    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    if user:
        return user

    try:
        return _create_user_with_team(
            db,
            firebase_uid=firebase_uid,
            email=decoded_token.get("email"),
            name=decoded_token.get("name"),
            avatar_url=decoded_token.get("picture"),
        )
    except IntegrityError:
        db.rollback()
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        if user:
            return user
        logger.exception("user provisioning failed for %s", firebase_uid)
        raise