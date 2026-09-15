from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.limiter import limiter
from app.deps import get_current_user
from app.models.user import User
from app.schemas.auth import EmailLinkRequest, EmailLinkResponse, MeResponse
from app.services.auth_links import send_sign_in_link, SignInLinkRateLimitedError

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=MeResponse)
@limiter.limit("30/minute")
def get_me(request: Request, user: User = Depends(get_current_user)):
    return user


@router.post("/authmail", response_model=EmailLinkResponse)
@limiter.limit("30/minute")
def send_authmail(request: Request, payload: EmailLinkRequest):
    try:
        send_sign_in_link(payload.email, payload.continue_url)
    except SignInLinkRateLimitedError as e:
        raise HTTPException(status_code=429, detail=str(e))
    return EmailLinkResponse(sent=True)
