import json

from fastapi import APIRouter, Depends, HTTPException, Request, File, Form, UploadFile
from sqlalchemy.orm import Session
from pydantic import BaseModel
from uuid import UUID
from typing import List

from app.core.database import get_db
from app.core.fal_webhook import FalWebhookVerificationError, verify_fal_webhook
from app.core.limiter import limiter
from app.deps import get_current_user
from app.models.user import User
from app.services.teams import is_team_member
from app.services.generation import create_generation_batch, fail_and_release, handle_fal_webhook
from app.models.generation_job import GenerationJob
from app.models.tool_definition import ToolDefinition
from app.core.arq_pool import get_arq_pool
from app.core.fal_client import upload_image_to_fal

router = APIRouter(tags=["generation"])


class GenerateRequest(BaseModel):
    team_id: UUID
    feature_type: str
    input_params: dict = {}
    output_count: int = 1


@router.post("/generate")
@limiter.limit("20/minute")
async def generate(
    request: Request,
    team_id: UUID = Form(...),
    feature_type: str = Form(...),
    color: str = Form(None),
    quality: str = Form(None),
    size: str = Form(None),
    target_area: str = Form(None),
    output_count: int = Form(1),
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_member(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    tool_check = db.query(ToolDefinition).filter(
        ToolDefinition.feature_type == feature_type
    ).first()
    if not tool_check:
        raise HTTPException(status_code=400, detail="Unknown tool")

    if len(images) > tool_check.max_input_images:
        raise HTTPException(
            status_code=400,
            detail=f"This tool accepts at most {tool_check.max_input_images} image(s)",
        )

    image_urls = []
    for image in images:
        file_bytes = await image.read()
        image_urls.append(upload_image_to_fal(file_bytes, image.filename, image.content_type))

    input_params = {
        "color": color,
        "quality": quality,
        "size": size,
        "target_area": target_area,
        "image_urls": image_urls,  # always a list now, even for single-image tools
    }
    input_params = {k: v for k, v in input_params.items() if v is not None}

    try:
        jobs = create_generation_batch(db, team_id, user.id, feature_type, input_params, output_count)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        pool = await get_arq_pool()
        for job in jobs:
            await pool.enqueue_job("submit_generation_to_fal", str(job.id))
    except Exception:
        for job in jobs:
            fail_and_release(db, job, "Failed to enqueue generation for processing")
        raise HTTPException(status_code=503, detail="Failed to start generation. Please try again.")

    return {
        "batchId": str(jobs[0].batch_id),
        "jobs": [{"jobId": str(j.id), "status": j.status} for j in jobs],
    }
@router.get("/jobs/{job_id}")
@limiter.limit("60/minute")
def get_job(
    job_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    job = db.query(GenerationJob).filter(GenerationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not is_team_member(db, job.team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    return {
        "jobId": str(job.id),
        "status": job.status,
        "outputUrl": job.output_url,
        "errorMessage": job.error_message,
        "creditsCharged": job.credits_charged,
    }


@router.get("/batches/{batch_id}")
@limiter.limit("60/minute")
def get_batch(
    batch_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    jobs = db.query(GenerationJob).filter(GenerationJob.batch_id == batch_id).all()
    if not jobs:
        raise HTTPException(status_code=404, detail="Batch not found")

    if not is_team_member(db, jobs[0].team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    return {
        "batchId": str(batch_id),
        "jobs": [
            {"jobId": str(j.id), "status": j.status, "outputUrl": j.output_url, "errorMessage": j.error_message}
            for j in jobs
        ],
    }


@router.post("/webhooks/fal")
async def fal_webhook(request: Request, job_id: UUID, db: Session = Depends(get_db)):
    raw_body = await request.body()

    try:
        verify_fal_webhook(request.headers, raw_body)
    except FalWebhookVerificationError as e:
        raise HTTPException(status_code=401, detail=str(e))

    payload = json.loads(raw_body)
    handle_fal_webhook(db, job_id, payload)
    return {"status": "ok"}

def validate_input_params(param_schema: list, input_params: dict) -> None:
    """
    Checks the incoming request against this tool's own param_schema —
    catches a missing required field or an invalid select option BEFORE
    any credits are charged or any job is created.
    """
    for field in param_schema:
        name = field["name"]
        value = input_params.get(name)

        if field.get("required") and not value:
            raise ValueError(f"Missing required field: {field['label']}")

        if field["type"] == "select" and value:
            options = field.get("options", [])
            # options can be a list of plain strings, OR a list of dicts
            # with a "value" key (used when each option carries its own
            # credit_cost, like Recolor's quality tiers)
            valid_values = [
                opt["value"] if isinstance(opt, dict) else opt
                for opt in options
            ]
            if value not in valid_values:
                raise ValueError(f"Invalid value for {field['label']}: {value}")