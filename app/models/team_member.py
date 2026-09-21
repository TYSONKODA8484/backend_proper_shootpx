from sqlalchemy import Column, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class TeamMember(Base):
    __tablename__ = "team_members"
    __table_args__ = (
        # Two concurrent invite-accepts for the same user must not be able to
        # both slip past the app-level "is this user already a member?" check
        # and create two rows -- see accept_invite()'s IntegrityError handling.
        UniqueConstraint("team_id", "user_id", name="uq_team_members_team_id_user_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(Text, nullable=False, default="owner")
    joined_at = Column(DateTime(timezone=True), server_default=func.now())