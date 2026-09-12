from fastapi import APIRouter, Request, HTTPException, Header, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.webhooks import (
    verify_webhook_signature,
    handle_payment_captured,
    handle_subscription_activated,
    handle_subscription_charged,
    handle_subscription_halted,
    handle_subscription_cancelled,
    handle_subscription_pending,
    handle_subscription_completed
)
router = APIRouter(prefix="/billing", tags=["webhooks"])


@router.post("/webhook")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str = Header(...),
    db: Session = Depends(get_db),
):
    raw_body = await request.body()

    if not verify_webhook_signature(raw_body, x_razorpay_signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    event = await request.json()

    event_type = event.get("event")

    if event_type == "payment.captured":
        handle_payment_captured(db, event)
    elif event_type == "subscription.activated":
        handle_subscription_activated(db, event)
    elif event_type == "subscription.charged":
        handle_subscription_charged(db, event)
    elif event_type == "subscription.halted":
        handle_subscription_halted(db, event)
    elif event_type == "subscription.cancelled":
        handle_subscription_cancelled(db, event)
    elif event_type == "subscription.pending":
        handle_subscription_pending(db, event)
    elif event_type == "subscription.completed":
        handle_subscription_completed(db, event)

    return {"status": "ok"}