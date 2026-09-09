import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import get_db
from app.core.cache import get_cached, set_cached
from app.models.tool import Tool
from app.schemas.tools import ToolsResponse, ToolOut
from app.core.limiter import limiter

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