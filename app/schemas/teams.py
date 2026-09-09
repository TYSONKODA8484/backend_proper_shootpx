from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from uuid import UUID


class TeamBillingOut(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    total_credits: int
    subscription_credits: int
    topup_credits: int
    plan: str | None = None  # subscription plan slug, e.g. "monthly-pro"; None when no subscription
    subscription_status: str | None = None
    current_period_end: str | None = None