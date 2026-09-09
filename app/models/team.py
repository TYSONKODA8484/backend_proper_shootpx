from sqlalchemy import Column, Text, DateTime, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    subscription_credits_remaining = Column(Integer, nullable=False, default=0)
    topup_credits_balance = Column(Integer, nullable=False, default=0)