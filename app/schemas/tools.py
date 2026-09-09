from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from uuid import UUID


class _CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class ToolOut(_CamelModel):
    id: UUID
    slug: str
    name: str
    description: str
    category: str
    status: str
    sort_order: int = 0


class ToolsResponse(_CamelModel):
    tools: list[ToolOut]
