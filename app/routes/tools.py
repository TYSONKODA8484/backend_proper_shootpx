import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import get_db
from app.core.cache import get_cached, set_cached
from app.models.tool_definition import ToolDefinition
from app.schemas.tools import (
    ToolsResponse, ToolOut, HomepageSlidesResponse, HomepageSlideOut,
)
from app.services.tool_definitions import get_tool_definition
from app.services.homepage import list_active_homepage_slides
from app.core.limiter import limiter
from app.deps import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/landing", tags=["tools"])
CACHE_KEY = "landing:tools"
SLIDES_CACHE_KEY = "landing:homepage-slides"

# These three endpoints are public, identical for every caller, and backed by
# hand-edited reference data -- exactly what a shared/CDN cache is for. Set on
# the RESPONSE (not just in our Redis layer) so every client benefits the same
# way: this frontend, a future mobile app, and any CDN put in front of the API.
# stale-while-revalidate lets a CDN serve the slightly-old copy instantly while
# it refreshes in the background, so nobody ever waits on our origin.
_CATALOG_CACHE_CONTROL = "public, max-age=60, s-maxage=300, stale-while-revalidate=600"
# Slides/cards are schedule-driven (starts_at/ends_at), so they get a shorter
# window -- a long cache could keep showing an expired slide or hide one that
# just went live.
_SCHEDULED_CACHE_CONTROL = "public, max-age=30, s-maxage=60, stale-while-revalidate=300"


@router.get("/tools", response_model=ToolsResponse)
@limiter.limit("30/minute")
def get_tools(request: Request, response: Response, db: Session = Depends(get_db)):
    response.headers["Cache-Control"] = _CATALOG_CACHE_CONTROL

    cached = get_cached(CACHE_KEY)
    if cached:
        return cached

    try:
        # Only tools with real marketing copy (display_name) are shown on the
        # public tool grid -- feature_types like enhance_prompt and
        # model_shoot_generate_model are internal/utility tools with no
        # display_name set, deliberately excluded rather than shown with
        # fabricated copy.
        tools = db.execute(
            select(ToolDefinition)
            .where(ToolDefinition.display_name.is_not(None))
            .order_by(ToolDefinition.category, ToolDefinition.card_sort_order)
        ).scalars().all()
    except SQLAlchemyError:
        logger.exception("tools query failed")
        raise HTTPException(status_code=503, detail="Tools data is unavailable")

    response = ToolsResponse(tools=[ToolOut.from_row(t) for t in tools])

    set_cached(CACHE_KEY, response.model_dump(mode="json", by_alias=True))
    return response


@router.get("/homepage-slides", response_model=HomepageSlidesResponse)
@limiter.limit("30/minute")
def get_homepage_slides(request: Request, response: Response, db: Session = Depends(get_db)):
    response.headers["Cache-Control"] = _SCHEDULED_CACHE_CONTROL

    cached = get_cached(SLIDES_CACHE_KEY)
    if cached:
        return cached

    try:
        slides = list_active_homepage_slides(db)
    except SQLAlchemyError:
        logger.exception("homepage slides query failed")
        raise HTTPException(status_code=503, detail="Homepage slides are unavailable")

    response = HomepageSlidesResponse(slides=[HomepageSlideOut.model_validate(s) for s in slides])

    # Short TTL (unlike the 1hr default) -- slides are schedule-driven
    # (starts_at/ends_at), so a long cache could keep showing an expired
    # slide or hide one that just went live.
    set_cached(SLIDES_CACHE_KEY, response.model_dump(mode="json", by_alias=True), ttl=60)
    return response


# --- Tool schema endpoint (backend config, used to build the generate form) ---
# Separate router: this is NOT a /landing route — it needs real auth, and it
# deliberately exposes only a filtered subset of tool_definitions.
tool_config_router = APIRouter(prefix="/tools", tags=["tool-config"])


@tool_config_router.get("/{feature_type}/schema")
@limiter.limit("60/minute")
def get_tool_schema(
    feature_type: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    tool = get_tool_definition(db, feature_type)

    if not tool or tool.stage != 1 or not tool.is_active:
        raise HTTPException(status_code=404, detail="Tool not found")

    # Deliberately an allowlist — fal_model_id and ai_steps must never reach
    # the frontend, regardless of what this row contains.
    return {
        "featureType": tool.feature_type,
        "maxInputImages": tool.max_input_images,
        "paramSchema": tool.param_schema,
        # The backend's own time budget for one generation of this tool. The
        # frontend uses it for "taking longer than expected" thresholds
        # instead of hardcoding a per-tool table that would silently drift
        # whenever a row in tool_definitions is edited. Not a hard client
        # deadline: a job can occasionally finish shortly after it.
        "generationTimeoutSeconds": tool.generation_timeout_seconds,
    }
