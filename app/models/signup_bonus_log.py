from sqlalchemy import Column, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class SignupBonusLog(Base):
    __tablename__ = "signup_bonus_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    ip_hash = Column(Text, nullable=False)
    device_id = Column(Text, nullable=True)
    bonus_granted = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
