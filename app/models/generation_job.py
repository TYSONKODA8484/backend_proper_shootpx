from sqlalchemy import Column, Text, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class GenerationJob(Base):
    __tablename__ = "generation_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    feature_type = Column(Text, nullable=False)
    stage = Column(Integer, nullable=False, default=1)
    batch_id = Column(UUID(as_uuid=True), nullable=True)
    source_job_id = Column(UUID(as_uuid=True), ForeignKey("generation_jobs.id"), nullable=True)
    status = Column(Text, nullable=False, default="queued")
    input_params = Column(JSONB, nullable=False, default=dict)
    output_url = Column(Text, nullable=True)
    fal_request_id = Column(Text, nullable=True)
    credits_charged = Column(Integer, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    credits_from_subscription = Column(Integer, nullable=False, default=0)
    credits_from_topup = Column(Integer, nullable=False, default=0)
    duration_seconds = Column(Integer, nullable=True)