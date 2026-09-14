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
from app.services.generation import create_generation_batch, fail_and_release, handle_fal_webhook, validate_input_params
from app.models.generation_job import GenerationJob
from app.models.tool_definition import ToolDefinition
from app.core.arq_pool import get_arq_pool
from app.core.fal_client import upload_image_to_fal

# 10 MB/image is generous for a product photo while still bounding memory use
# (await image.read() loads the whole file) and fal upload cost for obviously
# bad input.
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}

# output_count was previously bounded only by the team's credit balance, not
# by any sane per-request cap -- a typo (an extra zero) could create an
# enormous batch (thousands of DB rows + arq jobs) in one call for any team
# with enough credits. This is a request-shape limit, independent of cost.
MAX_OUTPUT_COUNT = 20

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

    if output_count > MAX_OUTPUT_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"output_count cannot exceed {MAX_OUTPUT_COUNT} per request",
        )

    # Validate the tool's own required/select fields (e.g. a missing color)
    # BEFORE paying to upload anything to fal -- a request that's going to be
    # rejected here shouldn't cost real fal upload quota every time.
    try:
        validate_input_params(
            tool_check.param_schema,
            {"color": color, "quality": quality, "size": size, "target_area": target_area},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Read + validate every image BEFORE uploading any of them -- same
    # reasoning: don't pay fal upload cost for a request that's about to be
    # rejected anyway (disallowed type, oversized file).
    read_images = []
    for image in images:
        if image.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type: {image.content_type}. "
                       f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_CONTENT_TYPES))}",
            )
        file_bytes = await image.read()
        if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"{image.filename} is over the {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB limit per image",
            )
        read_images.append((file_bytes, image.filename, image.content_type))

    image_urls = [
        upload_image_to_fal(file_bytes, filename, content_type)
        for file_bytes, filename, content_type in read_images
    ]

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
@limiter.limit("120/minute")
async def fal_webhook(request: Request, job_id: UUID, db: Session = Depends(get_db)):
    raw_body = await request.body()

    try:
        verify_fal_webhook(request.headers, raw_body)
    except FalWebhookVerificationError as e:
        raise HTTPException(status_code=401, detail=str(e))

    payload = json.loads(raw_body)
    handle_fal_webhook(db, job_id, payload)
    return {"status": "ok"}