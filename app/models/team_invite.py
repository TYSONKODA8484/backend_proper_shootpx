from sqlalchemy import Column, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class TeamInvite(Base):
    __tablename__ = "team_invites"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    email = Column(Text, nullable=False)
    role = Column(Text, nullable=False, default="editor")
    token = Column(Text, nullable=False, unique=True)
    status = Column(Text, nullable=False, default="pending")
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Set explicitly at creation (app/services/team_invites.py's
    # INVITE_EXPIRY_DAYS), not a DB default -- expiry is a computed check
    # (expires_at <= now), never its own status transition, so an invite
    # past this stays "pending" in the DB but is treated as gone by
    # create_invite's lookup, team_seat_count, list_pending_invites, and
    # accept_invite alike.
    expires_at = Column(DateTime(timezone=True), nullable=False)