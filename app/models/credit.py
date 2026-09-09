from sqlalchemy import Column, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from app.core.database import Base


class Credit(Base):
    __tablename__ = "credit"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug = Column(Text, nullable=False, unique=True)
    name = Column(Text, nullable=False)
    price = Column(Integer, nullable=False)
    credits = Column(Integer, nullable=False)
    info = Column(JSONB, nullable=False, default=list)
    tag = Column(Text, nullable=True)
    sort_order = Column(Integer, nullable=False, default=0)