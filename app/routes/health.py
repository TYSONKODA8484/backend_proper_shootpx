from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/")
def root():
    return {"service": settings.app_name, "status": "ok"}


@router.get("/health")
def health():
    return {"status": "healthy"}
