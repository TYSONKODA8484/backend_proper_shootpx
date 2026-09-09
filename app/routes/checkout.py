from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.limiter import limiter
from app.deps import get_current_user
from app.models.user import User
from app.services.teams import is_team_owner
from app.services.billing import create_credit_pack_checkout

router = APIRouter(prefix="/billing", tags=["checkout"])


@router.post("/teams/{team_id}/credit-packs/{pack_id}/checkout")
@limiter.limit("10/minute")
def checkout_credit_pack(
    request: Request,
    team_id: UUID,
    pack_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_owner(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="Only the team owner can buy credits")

    try:
        checkout = create_credit_pack_checkout(db, team_id, pack_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return checkout