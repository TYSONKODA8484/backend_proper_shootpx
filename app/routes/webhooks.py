from fastapi import APIRouter, Request, HTTPException, Header, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.webhooks import verify_webhook_signature, handle_payment_captured

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

    if event.get("event") == "payment.captured":
        handle_payment_captured(db, event)

    return {"status": "ok"}