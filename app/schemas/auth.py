from uuid import UUID

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class _CamelModel(BaseModel):
    """Base for responses: fields are snake_case in Python, camelCase in JSON."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class MeResponse(_CamelModel):
    id: UUID
    email: str
    name: str | None = None
    avatar_url: str | None = None


class EmailLinkRequest(BaseModel):
    email: str
    continue_url: str


class EmailLinkResponse(BaseModel):
    sent: bool
