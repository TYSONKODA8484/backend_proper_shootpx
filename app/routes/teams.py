from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.limiter import limiter
from app.deps import get_current_user
from app.models.user import User
from app.services.teams import (
    delete_team,
    get_team,
    is_team_member,
    is_team_owner,
    list_team_members,
    list_user_teams,
    remove_member,
    rename_team,
    get_team_billing
)
from app.schemas.teams import TeamBillingOut
from app.services.team_invites import accept_invite, create_invite, TeamFullError

router = APIRouter(tags=["teams"])


# --- request bodies --------------------------------------------------------

class InviteRequest(BaseModel):
    email: str
    role: Literal["editor", "owner"] = "editor"


class RenameTeamRequest(BaseModel):
    name: str


def _require_owner(db: Session, team_id: UUID, user: User) -> None:
    if not is_team_owner(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="Only the team owner can do this")


# --- reads ---------------------------------------------------------------

@router.get("/teams")
@limiter.limit("20/minute")
def my_teams(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return {"teams": list_user_teams(db, user.id)}


@router.get("/teams/{team_id}/members")
@limiter.limit("30/minute")
def team_members(
    request: Request,
    team_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_member(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    team = get_team(db, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    return {
        "id": str(team.id),
        "name": team.name,
        "members": list_team_members(db, team_id),
    }


# --- invites -----------------------------------------------------------

@router.post("/teams/{team_id}/invite")
@limiter.limit("10/minute")
def invite_to_team(
    request: Request,
    team_id: UUID,
    payload: InviteRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_owner(db, team_id, user)

    try:
        invite = create_invite(db, team_id, payload.email, user.id, payload.role)
    except TeamFullError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"sent": True, "email": invite.email, "role": invite.role}


@router.post("/invites/{token}/accept")
@limiter.limit("10/minute")
def accept_team_invite(
    request: Request,
    token: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        invite = accept_invite(db, token, user.id, user.email)
    except TeamFullError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"joinedTeamId": str(invite.team_id), "role": invite.role}


# --- owner management -------------------------------------------------

@router.patch("/teams/{team_id}")
@limiter.limit("10/minute")
def patch_team(
    request: Request,
    team_id: UUID,
    payload: RenameTeamRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_owner(db, team_id, user)

    team = get_team(db, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    try:
        rename_team(db, team, payload.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"id": str(team.id), "name": team.name}


@router.delete("/teams/{team_id}/members/{member_user_id}")
@limiter.limit("10/minute")
def remove_team_member(
    request: Request,
    team_id: UUID,
    member_user_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_owner(db, team_id, user)

    try:
        remove_member(db, team_id, member_user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"removed": str(member_user_id)}


@router.delete("/teams/{team_id}")
@limiter.limit("5/minute")
def delete_team_route(
    request: Request,
    team_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_owner(db, team_id, user)

    if not get_team(db, team_id):
        raise HTTPException(status_code=404, detail="Team not found")

    delete_team(db, team_id)
    return {"deleted": str(team_id)}

@router.get("/teams/{team_id}/billing", response_model=TeamBillingOut)
@limiter.limit("30/minute")
def team_billing(
    request: Request,
    team_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_member(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    try:
        billing = get_team_billing(db, team_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return billing