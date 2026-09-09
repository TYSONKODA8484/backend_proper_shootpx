from sqlalchemy import Column, Text, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class TeamSubscription(Base):
    __tablename__ = "team_subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False, unique=True)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscription.id"), nullable=False)
    razorpay_subscription_id = Column(Text, nullable=True)
    status = Column(Text, nullable=False, default="active")
    credits_per_refill = Column(Integer, nullable=False)
    next_refill_at = Column(DateTime(timezone=True), nullable=False)
    current_period_end = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())