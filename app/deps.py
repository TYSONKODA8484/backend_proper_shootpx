"""Shared FastAPI dependencies."""

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import get_db
from app.core.firebase import verify_token
from app.models.user import User
from app.services.users import get_or_create_user


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
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    token = _bearer_token(authorization)

    try:
        decoded = verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    try:
        return get_or_create_user(db, decoded)
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Could not load user")
