import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.models.generation_job import GenerationJob
from app.models.tool_definition import ToolDefinition
from app.services.generation_lock import acquire_generation_lock, release_generation_lock, release_fal_slot
from app.services.credits import spend_credits, refund_credits
from app.core.storage import upload_to_storage, download_from_url

logger = logging.getLogger(__name__)

DOWNLOAD_FAILED_MESSAGE = "Could not retrieve the generated image. Please try again."
UPLOAD_FAILED_MESSAGE = "Could not save the generated image. Please try again."


def validate_input_params(param_schema: list, input_params: dict) -> None:
    for field in param_schema:
        name = field["name"]
        value = input_params.get(name)

        if field.get("required") and not value:
            raise ValueError(f"Missing required field: {field['label']}")

        if field["type"] == "select" and value:
            options = field.get("options", [])
            valid_values = [
                opt["value"] if isinstance(opt, dict) else opt
                for opt in options
            ]
            if value not in valid_values:
                raise ValueError(f"Invalid value for {field['label']}: {value}")


def _resolve_credit_cost(param_schema: list, input_params: dict, fallback_cost: int) -> int:
    for field in param_schema:
        if field["type"] != "select":
            continue
        options = field.get("options", [])
        if not options or not isinstance(options[0], dict):
            continue
        if "credit_cost" not in options[0]:
            continue

        selected_value = input_params.get(field["name"])
        for opt in options:
            if opt["value"] == selected_value:
                return opt["credit_cost"]

    return fallback_cost


def create_generation_batch(db: Session, team_id, user_id, feature_type: str, input_params: dict, output_count: int = 1) -> list[GenerationJob]:
    if not acquire_generation_lock(user_id):
        raise ValueError("You already have a generation in progress. Please wait for it to finish.")

    try:
        tool = db.query(ToolDefinition).filter(
            ToolDefinition.feature_type == feature_type
        ).first()
        if not tool:
            raise ValueError("Unknown tool")
        if not tool.is_active:
            raise ValueError("This tool is not currently available")

        validate_input_params(tool.param_schema, input_params)

        output_count = output_count if output_count and output_count > 0 else tool.default_output_count
        per_output_cost = _resolve_credit_cost(tool.param_schema, input_params, tool.credit_cost_per_output)
        total_cost = per_output_cost * output_count

        _, from_subscription, from_topup = spend_credits(db, team_id, total_cost, commit=False)

        per_job_sub = from_subscription // output_count
        per_job_top = from_topup // output_count
        remainder_sub = from_subscription - (per_job_sub * output_count)
        remainder_top = from_topup - (per_job_top * output_count)

        batch_id = uuid.uuid4()
        jobs = []

        for i in range(output_count):
            job = GenerationJob(
                team_id=team_id,
                user_id=user_id,
                feature_type=feature_type,
                input_params=input_params,
                batch_id=batch_id,
                credits_charged=per_output_cost,
                credits_from_subscription=per_job_sub + (remainder_sub if i == 0 else 0),
                credits_from_topup=per_job_top + (remainder_top if i == 0 else 0),
                status="queued",
            )
            db.add(job)
            jobs.append(job)

        db.commit()
        return jobs

    except Exception:
        db.rollback()
        release_generation_lock(user_id)
        raise


def fail_and_release(db: Session, job: GenerationJob, error_message: str) -> None:
    """
    Only used for the enqueue-failure path (/generate -> arq), before the job
    has ever reached the worker -- so no fal slot was ever reserved for it
    (that only happens inside submit_generation_to_fal). Must NOT call
    release_fal_slot here, or it decrements a counter nothing incremented.
    """
    job.status = "failed"
    job.error_message = error_message
    refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)
    job.completed_at = datetime.now(timezone.utc)

    remaining = db.query(GenerationJob).filter(
        GenerationJob.batch_id == job.batch_id,
        GenerationJob.status.in_(("queued", "processing")),
    ).count()

    if remaining <= 1:
        release_generation_lock(job.user_id)

    db.commit()


def handle_fal_webhook(db: Session, job_id, payload: dict):
    job = db.query(GenerationJob).filter(GenerationJob.id == job_id).with_for_update().first()
    if not job:
        return

    if job.status in ("completed", "failed"):
        return

    if payload.get("status") == "OK":
        images = payload.get("payload", {}).get("images", [])
        fal_url = images[0]["url"] if images else None

        if fal_url:
            try:
                file_bytes = download_from_url(fal_url)
            except Exception:
                logger.exception("Failed to download fal.ai output for job %s", job.id)
                job.status = "failed"
                job.error_message = DOWNLOAD_FAILED_MESSAGE
                refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)
            else:
                try:
                    permanent_path = f"{job.team_id}/{job.feature_type}/{job.id}/output.png"
                    job.output_url = upload_to_storage(permanent_path, file_bytes)
                    job.status = "completed"
                except Exception:
                    logger.exception("Failed to upload output to storage for job %s", job.id)
                    job.status = "failed"
                    job.error_message = UPLOAD_FAILED_MESSAGE
                    refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)
        else:
            job.status = "failed"
            job.error_message = "fal reported success but returned no image"
            refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)
    else:
        job.status = "failed"
        job.error_message = payload.get("error", "Generation failed")
        refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)

    job.completed_at = datetime.now(timezone.utc)

    remaining = db.query(GenerationJob).filter(
        GenerationJob.batch_id == job.batch_id,
        GenerationJob.status.in_(("queued", "processing")),
    ).count()

    if remaining <= 1:
        release_generation_lock(job.user_id)
        release_fal_slot(job.team_id)

    db.commit()