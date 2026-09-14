from sqlalchemy import Column, Text, Integer, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from app.core.database import Base


class ToolDefinition(Base):
    __tablename__ = "tool_definitions"

    feature_type = Column(Text, primary_key=True)
    fal_model_id = Column(Text, nullable=False)
    max_input_images = Column(Integer, nullable=False, default=1)
    max_output_resolution = Column(Text, nullable=False)
    default_output_count = Column(Integer, nullable=False, default=1)
    credit_cost_per_output = Column(Integer, nullable=False)
    param_schema = Column(JSONB, nullable=False, default=dict)
    is_active = Column(Boolean, nullable=False, default=True)
    ai_steps = Column(JSONB, nullable=False, default=dict)
    stage = Column(Integer, nullable=False, default=1)