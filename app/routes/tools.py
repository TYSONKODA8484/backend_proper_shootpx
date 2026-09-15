import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import get_db
from app.core.cache import get_cached, set_cached
from app.models.tool import Tool
from app.schemas.tools import ToolsResponse, ToolOut
from app.services.tool_definitions import get_tool_definition
from app.core.limiter import limiter
from app.deps import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/landing", tags=["tools"])
CACHE_KEY = "landing:tools"


@router.get("/tools", response_model=ToolsResponse)
@limiter.limit("30/minute")
def get_tools(request: Request, db: Session = Depends(get_db)):
    cached = get_cached(CACHE_KEY)
    if cached:
        return cached

    try:
        tools = db.execute(
            select(Tool).order_by(Tool.category, Tool.sort_order)
        ).scalars().all()
    except SQLAlchemyError:
        logger.exception("tools query failed")
        raise HTTPException(status_code=503, detail="Tools data is unavailable")

    response = ToolsResponse(tools=[ToolOut.model_validate(t) for t in tools])

    set_cached(CACHE_KEY, response.model_dump(mode="json", by_alias=True))
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

    # Deliberately return ONLY these two fields — fal_model_id and ai_steps
    # must never reach the frontend, regardless of what this row contains.
    return {
        "featureType": tool.feature_type,
        "maxInputImages": tool.max_input_images,
        "paramSchema": tool.param_schema,
    }