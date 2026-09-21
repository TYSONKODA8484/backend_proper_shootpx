from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.homepage_slide import HomepageSlide


def _active_now(model):
    now = datetime.now(timezone.utc)
    return (
        model.is_active == True,  # noqa: E712
        or_(model.starts_at.is_(None), model.starts_at <= now),
        or_(model.ends_at.is_(None), model.ends_at >= now),
    )


def list_active_homepage_slides(db: Session) -> list[HomepageSlide]:
    return (
        db.query(HomepageSlide)
        .filter(*_active_now(HomepageSlide))
        .order_by(HomepageSlide.sort_order)
        .all()
    )
