import logging
import secrets

from firebase_admin import auth as firebase_auth
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.email import send_email
from app.models.team import Team
from app.models.team_invite import TeamInvite
from app.models.team_member import TeamMember
from app.services.teams import MAX_TEAM_MEMBERS, team_member_count, team_seat_count

logger = logging.getLogger(__name__)


class TeamFullError(ValueError):
    """Raised when a team is already at MAX_TEAM_MEMBERS (members + pending invites)."""


def _invite_email_html(team_name: str, link: str, role: str) -> str:
    return f"""\
<div style="font-family:system-ui,Arial,sans-serif;max-width:480px;margin:0 auto">
  <h2 style="margin:0 0 12px">You've been invited to {team_name}</h2>
  <p style="color:#444;line-height:1.5">
    You've been invited to join the <strong>{team_name}</strong> team on ShootPX
    as <strong>{role}</strong>.
    Click the button below — it signs you in and adds you to the team.
  </p>
  <p style="margin:24px 0">
    <a href="{link}"
       style="background:#111;color:#fff;padding:10px 18px;border-radius:6px;text-decoration:none">
      Accept invitation
    </a>
  </p>
  <p style="color:#888;font-size:13px">
    If the button doesn't work, paste this link into your browser:<br>{link}
  </p>
  <p style="color:#aaa;font-size:12px">If you weren't expecting this, you can ignore this email.</p>
</div>"""


ALLOWED_ROLES = ("editor", "owner")


def create_invite(
    db: Session, team_id, email: str, invited_by_user_id, role: str = "editor"
) -> TeamInvite:
    email = email.strip().lower()
    if role not in ALLOWED_ROLES:
        raise ValueError(f"role must be one of {ALLOWED_ROLES}")

    invite = (
        db.query(TeamInvite)
        .filter(
            TeamInvite.team_id == team_id,
            TeamInvite.email == email,
            TeamInvite.status == "pending",
        )
        .first()
    )

    if invite is None:
        # a brand-new invite takes a seat — block if the team is already full
        if team_seat_count(db, team_id) >= MAX_TEAM_MEMBERS:
            raise TeamFullError(
                f"Team is full — max {MAX_TEAM_MEMBERS} members (owner + editors, "
                "pending invites included)"
            )
        invite = TeamInvite(
            team_id=team_id,
            email=email,
            role=role,
            token=secrets.token_urlsafe(24),
            invited_by=invited_by_user_id,
        )
        db.add(invite)
        db.commit()
    elif invite.role != role:
        # re-inviting with a different role — update the pending invite
        invite.role = role
        db.commit()

    team = db.query(Team).filter(Team.id == team_id).first()
    team_name = team.name if team else "a ShootPX team"

    # The link is a Firebase email sign-in link. Clicking it signs the invitee in
    # AS THIS EXACT EMAIL, then lands on the console with the invite token so the
    # page can accept it automatically.
    continue_url = (
        f"{settings.frontend_url}/testconsole.html"
        f"?invite_token={invite.token}&invite_email={email}"
    )
    action_code_settings = firebase_auth.ActionCodeSettings(
        url=continue_url, handle_code_in_app=True
    )
    link = firebase_auth.generate_sign_in_with_email_link(email, action_code_settings)

    send_email(
        to=email,
        subject=f"Invitation to join {team_name} on ShootPX",
        html=_invite_email_html(team_name, link, invite.role),
        text=(
            f"You've been invited to join {team_name} on ShootPX as {invite.role}. "
            f"Accept here: {link}"
        ),
    )

    return invite


def accept_invite(db: Session, token: str, user_id, user_email: str) -> TeamInvite:
    invite = db.query(TeamInvite).filter(TeamInvite.token == token).first()

    if not invite:
        raise ValueError("Invite not found")
    if invite.status != "pending":
        raise ValueError("Invite already used")
    if invite.email.strip().lower() != user_email.strip().lower():
        raise ValueError("This invite was sent to a different email address")

    member = (
        db.query(TeamMember)
        .filter(
            TeamMember.team_id == invite.team_id,
            TeamMember.user_id == user_id,
        )
        .first()
    )
    if member is None:
        # joining as a new member takes a seat — block if the team filled up
        # since the invite was created
        if team_member_count(db, invite.team_id) >= MAX_TEAM_MEMBERS:
            raise TeamFullError(f"Team is full — max {MAX_TEAM_MEMBERS} members")
        db.add(TeamMember(team_id=invite.team_id, user_id=user_id, role=invite.role))
    elif member.role != invite.role:
        # already on the team — apply the role from the invite
        member.role = invite.role

    invite.status = "accepted"
    db.commit()

    return invite
