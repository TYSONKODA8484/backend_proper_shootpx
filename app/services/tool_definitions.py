import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.cache import get_cached, set_cached
from app.models.tool_definition import ToolDefinition

logger = logging.getLogger(__name__)

CACHE_KEY_PREFIX = "tooldef:"

# tool_definitions rows are edited directly via SQL (there's no admin UI /
# migration for them), so there is no code path that can proactively
# invalidate a specific row the moment it changes. A short TTL bounds
# worst-case staleness on its own; POST /admin/cache/clear (which clears
# "tooldef:*" alongside "landing:*") gives an immediate way to force a
# refresh right after making an edit, instead of waiting out the TTL.
CACHE_TTL_SECONDS = 60


@dataclass(frozen=True)
class CachedToolDefinition:
    """
    Plain, attribute-compatible stand-in for the ToolDefinition ORM row --
    every field every caller actually reads (tool.ai_steps, tool.is_active,
    etc.), reconstructed from the cached JSON so existing `tool.<attr>` call
    sites (worker.py, generation.py, routes, enhance_prompt.py) don't need to
    change at all. Read-only: nothing in this codebase ever writes to a
    ToolDefinition object it queried, only reads from it.
    """
    feature_type: str
    fal_model_id: str
    max_input_images: int
    max_output_resolution: str
    default_output_count: int
    credit_cost_per_output: int
    param_schema: list
    is_active: bool
    stage: int
    ai_steps: dict
    generation_timeout_seconds: int


def _to_cache_dict(tool: ToolDefinition) -> dict:
    return {
        "feature_type": tool.feature_type,
        "fal_model_id": tool.fal_model_id,
        "max_input_images": tool.max_input_images,
        "max_output_resolution": tool.max_output_resolution,
        "default_output_count": tool.default_output_count,
        "credit_cost_per_output": tool.credit_cost_per_output,
        "param_schema": tool.param_schema,
        "is_active": tool.is_active,
        "stage": tool.stage,
        "ai_steps": tool.ai_steps,
        "generation_timeout_seconds": tool.generation_timeout_seconds,
    }


def get_tool_definition(db: Session, feature_type: str):
    """
    Cached lookup by feature_type -- returns a CachedToolDefinition (cache
    hit) or the live ToolDefinition ORM row (cache miss, freshly queried and
    now cached), or None if no such tool exists. Callers only ever read
    attributes, so both return types are interchangeable in practice.
    """
    if not feature_type:
        # Defensive: every real caller only reaches here with a non-empty
        # string (enforced upstream by param_schema's `required` flag on
        # every tool that needs one, e.g. enhance_prompt's
        # source_feature_type) -- but this is the boundary function every
        # such caller shares, so it must fail as a clean "no such tool"
        # lookup rather than crash with a raw TypeError from
        # `CACHE_KEY_PREFIX + None` the moment any caller's assumption ever
        # breaks.
        return None

    cache_key = CACHE_KEY_PREFIX + feature_type
    cached = get_cached(cache_key)
    if cached is not None:
        return CachedToolDefinition(**cached)

    tool = db.query(ToolDefinition).filter(ToolDefinition.feature_type == feature_type).first()
    if tool is None:
        return None

    set_cached(cache_key, _to_cache_dict(tool), ttl=CACHE_TTL_SECONDS)
    return tool
