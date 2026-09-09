"""Aggregates every route module into a single router for `main` to include."""

from fastapi import APIRouter

from app.routes import auth, billing, cache, health, tools
from app.routes.teams import router as teams_router
from app.routes.checkout import router as checkout_router
from app.routes.webhooks import router as webhooks_router

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(billing.router)
api_router.include_router(tools.router)
api_router.include_router(cache.router)
api_router.include_router(teams_router)
api_router.include_router(checkout_router)
api_router.include_router(webhooks_router)

