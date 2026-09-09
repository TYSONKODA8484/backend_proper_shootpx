from sqlalchemy.orm import Session

from app.core.razorpay_client import razorpay_client
from app.models.credit import Credit
from app.core.config import settings


def create_credit_pack_checkout(db: Session, team_id, pack_id) -> dict:
    pack = db.query(Credit).filter(Credit.id == pack_id).first()
    if not pack:
        raise ValueError("Credit pack not found")

    order = razorpay_client.order.create({
        "amount": pack.price,       # already in paise
        "currency": "INR",
        "notes": {
            # Razorpay note values must be strings — the webhook parses `credits`
            # back with int().
            "team_id": str(team_id),
            "credit_pack_id": str(pack_id),
            "credits": str(pack.credits),
        },
    })

    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "key_id": settings.razorpay_key_id,
    }