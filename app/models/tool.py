from sqlalchemy import Column, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
import uuid
from app.core.database import Base


class Tool(Base):
    __tablename__ = "tools"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug = Column(Text, nullable=False, unique=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(Text, nullable=False)  # Shoot / Finish / Video / Scale
    status = Column(Text, nullable=False)    # live / soon
    sort_order = Column(Integer, nullable=False, default=0)
