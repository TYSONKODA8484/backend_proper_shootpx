import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.models.generation_job import GenerationJob
from app.models.tool_definition import ToolDefinition
from app.services.generation_lock import acquire_generation_lock, release_generation_lock
from app.services.credits import spend_credits, refund_credits
from app.core.storage import upload_to_storage, download_from_url

logger = logging.getLogger(__name__)

# job.error_message is returned verbatim to any team member via GET /jobs/{id}
# and GET /batches/{id} -- these must always be fixed, generic strings, never
# raw exception text (that lesson came from a real FAL_KEY leak found earlier
# in this pipeline's audit). The real exception detail is logged server-side
# via logger.exception() instead, distinguishing the two failure points so
# support/ops can tell a dead fal.ai temp URL apart from a broken storage
# bucket without ever exposing either to the client.
DOWNLOAD_FAILED_MESSAGE = "Could not retrieve the generated image. Please try again."
UPLOAD_FAILED_MESSAGE = "Could not save the generated image. Please try again."


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

        output_count = output_count if output_count and output_count > 0 else tool.default_output_count
        total_cost = tool.credit_cost_per_output * output_count

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
                credits_charged=tool.credit_cost_per_output,
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
    Used when a job was created and charged, but something after that point
    failed before fal.ai was ever reached (e.g. arq/Redis unreachable at
    enqueue time). Marks the job failed, refunds using its own exact stored
    split, releases the lock if this was the last job in its batch.
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

    db.commit()