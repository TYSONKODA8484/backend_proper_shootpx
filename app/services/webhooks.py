import hmac
import hashlib

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.models.billing_transaction import BillingTransaction
from app.services.credits import add_topup_credits


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    expected = hmac.new(
        settings.razorpay_webhook_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def handle_payment_captured(db: Session, event: dict):
    payment = event["payload"]["payment"]["entity"]
    razorpay_payment_id = payment["id"]
    order_notes = payment.get("notes", {})

    # Fast path: a retried webhook whose original already committed.
    existing = db.query(BillingTransaction).filter(
        BillingTransaction.razorpay_payment_id == razorpay_payment_id
    ).first()
    if existing:
        return  # already handled, do nothing — safe to ignore a retried webhook

    team_id = order_notes.get("team_id")
    credits = int(order_notes.get("credits", 0))
    amount = payment["amount"]

    if not team_id or not credits:
        return  # not a credit-pack payment we recognize, ignore safely

    # Claim this payment id and grant the credits in ONE transaction. The unique
    # constraint on razorpay_payment_id is the real idempotency guard: if a
    # concurrent (or racing retried) webhook already claimed it, the flush fails
    # and we bail out before touching the balance — no double-credit.
    db.add(BillingTransaction(
        team_id=team_id,
        type="credit_pack",
        razorpay_payment_id=razorpay_payment_id,
        amount=amount,
        credits_added=credits,
        status="completed",
    ))
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        return  # another worker got here first

    add_topup_credits(db, team_id, credits, commit=False)
    db.commit()