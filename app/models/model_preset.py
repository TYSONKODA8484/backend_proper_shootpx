from sqlalchemy import Column, Text, Boolean
from app.core.database import Base


class ModelPreset(Base):
    __tablename__ = "model_presets"

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)
    image_url = Column(Text, nullable=False)
    thumbnail_url = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
