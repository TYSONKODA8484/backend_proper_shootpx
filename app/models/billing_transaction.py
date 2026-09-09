from sqlalchemy import Column, Text, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class BillingTransaction(Base):
    __tablename__ = "billing_transactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    type = Column(Text, nullable=False)
    razorpay_payment_id = Column(Text, nullable=False, unique=True)
    amount = Column(Integer, nullable=False)
    credits_added = Column(Integer, nullable=False)
    status = Column(Text, nullable=False, default="completed")
    created_at = Column(DateTime(timezone=True), server_default=func.now())