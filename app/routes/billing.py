import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import get_db
from app.core.cache import get_cached, set_cached
from app.models.subscription import Subscription
from app.models.credit import Credit
from app.schemas.billing import BillingResponse, SubscriptionOut, CreditOut
from app.core.limiter import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/landing", tags=["billing"])
CACHE_KEY = "landing:billing"


@router.get("/billing", response_model=BillingResponse)
@limiter.limit("30/minute")
def get_billing(request: Request, db: Session = Depends(get_db)):
    cached = get_cached(CACHE_KEY)
    if cached:
        return cached

    try:
        subs = db.execute(
            select(Subscription).order_by(Subscription.sort_order)
        ).scalars().all()

        packs = db.execute(
            select(Credit).order_by(Credit.sort_order)
        ).scalars().all()
    except SQLAlchemyError:
        logger.exception("billing query failed")
        raise HTTPException(status_code=503, detail="Billing data is unavailable")

    response = BillingResponse(
        subscriptions=[SubscriptionOut.model_validate(s) for s in subs],
        credits=[CreditOut.model_validate(p) for p in packs],
    )

    set_cached(CACHE_KEY, response.model_dump(mode="json", by_alias=True))
    return response