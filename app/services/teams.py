from sqlalchemy.orm import Session

from app.models.team import Team
from app.models.team_invite import TeamInvite
from app.models.team_member import TeamMember
from app.models.user import User
from uuid import UUID

# Hard cap on team size: owner + editors. Pending invites also count toward it
# (each pending invite reserves a seat), so a team can never end up over the cap.
MAX_TEAM_MEMBERS = 5


# --- membership checks ---------------------------------------------------------

def _membership(db: Session, team_id, user_id) -> TeamMember | None:
    return (
        db.query(TeamMember)
        .filter(TeamMember.team_id == team_id, TeamMember.user_id == user_id)
        .first()
    )


def is_team_member(db: Session, team_id, user_id) -> bool:
    return _membership(db, team_id, user_id) is not None


def is_team_owner(db: Session, team_id, user_id) -> bool:
    m = _membership(db, team_id, user_id)
    return m is not None and m.role == "owner"


# --- reads --------------------------------------------------------------------

def get_team(db: Session, team_id) -> Team | None:
    return db.query(Team).filter(Team.id == team_id).first()


def list_user_teams(db: Session, user_id) -> list[dict]:
    rows = (
        db.query(Team, TeamMember.role)
        .join(TeamMember, TeamMember.team_id == Team.id)
        .filter(TeamMember.user_id == user_id)
        .order_by(TeamMember.joined_at)
        .all()
    )
    return [{"id": str(t.id), "name": t.name, "role": role} for t, role in rows]


def list_team_members(db: Session, team_id) -> list[dict]:
    rows = (
        db.query(TeamMember, User)
        .join(User, User.id == TeamMember.user_id)
        .filter(TeamMember.team_id == team_id)
        .order_by(TeamMember.joined_at)
        .all()
    )
    return [
        {
            "userId": str(u.id),
            "email": u.email,
            "name": u.name,
            "avatarUrl": u.avatar_url,
            "role": m.role,
            "joinedAt": m.joined_at.isoformat() if m.joined_at else None,
        }
        for m, u in rows
    ]


def team_member_count(db: Session, team_id) -> int:
    return db.query(TeamMember).filter(TeamMember.team_id == team_id).count()


def owner_count(db: Session, team_id) -> int:
    return (
        db.query(TeamMember)
        .filter(TeamMember.team_id == team_id, TeamMember.role == "owner")
        .count()
    )


def team_seat_count(db: Session, team_id) -> int:
    """Current members + still-pending invites."""
    pending = (
        db.query(TeamInvite)
        .filter(TeamInvite.team_id == team_id, TeamInvite.status == "pending")
        .count()
    )
    return team_member_count(db, team_id) + pending


# --- writes (all callers must have already checked the acting user is owner) ---

def rename_team(db: Session, team: Team, name: str) -> Team:
    name = name.strip()
    if not name:
        raise ValueError("Team name cannot be empty")
    team.name = name
    db.commit()
    return team


def remove_member(db: Session, team_id, target_user_id) -> None:
    member = _membership(db, team_id, target_user_id)
    if member is None:
        raise ValueError("That user is not a member of this team")
    if member.role == "owner" and owner_count(db, team_id) <= 1:
        raise ValueError("Cannot remove the last owner — transfer ownership or delete the team")

    db.delete(member)
    db.commit()


def delete_team(db: Session, team_id) -> None:
    db.query(TeamInvite).filter(TeamInvite.team_id == team_id).delete(synchronize_session=False)
    db.query(TeamMember).filter(TeamMember.team_id == team_id).delete(synchronize_session=False)
    db.query(Team).filter(Team.id == team_id).delete(synchronize_session=False)
    db.commit()

def get_team_billing(db: Session, team_id: UUID) -> dict:
    from app.models.subscription import Subscription
    from app.models.team_subscription import TeamSubscription
    from app.services.credits import get_total_credits

    team = get_team(db, team_id)
    if not team:
        raise ValueError("Team not found")

    row = (
        db.query(TeamSubscription, Subscription)
        .join(Subscription, Subscription.id == TeamSubscription.subscription_id)
        .filter(TeamSubscription.team_id == team_id)
        .first()
    )
    sub, plan = row if row else (None, None)

    return {
        "total_credits": get_total_credits(team),
        "subscription_credits": team.subscription_credits_remaining,
        "topup_credits": team.topup_credits_balance,
        "plan": plan.slug if plan else None,
        "subscription_status": sub.status if sub else None,
        "current_period_end": sub.current_period_end.isoformat() if sub else None,
    }