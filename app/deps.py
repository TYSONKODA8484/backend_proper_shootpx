"""Shared FastAPI dependencies."""

from fastapi import BackgroundTasks, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import get_db
from app.core.firebase import verify_token
from app.core.limiter import client_ip
from app.models.user import User
from app.services.users import CachedUser, get_or_create_user


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=401, detail="Authorization header must be 'Bearer <token>'"
        )
    return token.strip()


def get_current_user(
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(default=None),
    x_device_id: str | None = Header(default=None, alias="X-Device-Id"),
    db: Session = Depends(get_db),
) -> User | CachedUser:
    token = _bearer_token(authorization)

    try:
        decoded = verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    try:
        # ip/x_device_id only matter the one time this call turns out to be
        # a brand-new user (see get_or_create_user) -- there is no distinct
        # "signup" request in this API, account creation is implicit on
        # whichever authenticated endpoint the frontend happens to hit first
        # after a new Firebase sign-in, so the anti-abuse check has to live
        # at this single chokepoint every authed route already goes through.
        return get_or_create_user(
            db, decoded, ip=client_ip(request), device_id=x_device_id,
            background_tasks=background_tasks,
        )
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Could not load user")
