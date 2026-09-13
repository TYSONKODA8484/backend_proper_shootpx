from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.core.arq_pool import get_arq_pool
from app.core.config import settings
from app.models.generation_job import GenerationJob
from app.models.tool_definition import ToolDefinition
from app.services.generation_lock import acquire_generation_lock, release_generation_lock
from app.services.credits import spend_credits, refund_credits
from app.core.storage import upload_to_storage, download_from_url


def _redact_secrets(error: Exception) -> str:
    """job.error_message is returned to any team member via GET /jobs/{id};
    real fal.ai/Supabase exceptions never include these, but this is defense
    in depth against a future exception type that did."""
    text = str(error)
    for secret in (settings.fal_key, settings.supabase_service_role_key):
        text = text.replace(secret, "[REDACTED]")
    return text


def create_generation_job(db: Session, team_id, user_id, feature_type: str, input_params: dict) -> GenerationJob:
    # 1. Lock check — one generation at a time, per person
    if not acquire_generation_lock(user_id):
        raise ValueError("You already have a generation in progress. Please wait for it to finish.")

    try:
        # 2. Look up the tool's real definition (not the frontend display table)
        tool = db.query(ToolDefinition).filter(
            ToolDefinition.feature_type == feature_type
        ).first()
        if not tool:
            raise ValueError("Unknown tool")
        if not tool.is_active:
            raise ValueError("This tool is not currently available")

        # 3. Charge credits BEFORE calling fal.ai — this is the "reservation"
        #    from the credits design. If fal.ai fails, this gets refunded.
        #    commit=False: the spend and the job row below must land in the
        #    same transaction, otherwise a failure creating the job would
        #    leave credits spent with no job row to ever refund them from.
        _, from_subscription, from_topup = spend_credits(db, team_id, tool.credit_cost_per_output, commit=False)

        # 4. Create the job row — records exactly which pools were charged,
        #    so a later refund (if this job fails) is always exact, never guessed.
        job = GenerationJob(
            team_id=team_id,
            user_id=user_id,
            feature_type=feature_type,
            input_params=input_params,
            credits_charged=tool.credit_cost_per_output,
            credits_from_subscription=from_subscription,
            credits_from_topup=from_topup,
            status="queued",
        )
        db.add(job)
        db.commit()

        # 5. STUB — real fal.ai call comes in Part 2. For now, just simulate
        #    "submitted successfully" so the pipeline can be tested end to end.
        # TODO: replace with real fal.ai submit + webhook URL

        return job

    except Exception:
        # Roll back so a failed job insert can never leave the just-spent
        # credits committed with no job row to account for them.
        db.rollback()
        # If anything above fails (credit issue, tool lookup, etc.), release
        # the lock immediately — don't leave the person stuck for 5 minutes
        # over an error that happened before any real job started.
        release_generation_lock(user_id)
        raise


def fail_and_release(db: Session, job: GenerationJob, error_message: str) -> None:
    """Mark a job failed, refund its exact charged split, and release the
    lock. Used when a job could not even be enqueued for processing (e.g.
    arq/Redis unreachable right after /generate created it) so it's never
    left stuck 'queued' forever with credits already spent, a lock held, and
    nothing ever going to process it."""
    job.status = "failed"
    job.error_message = error_message
    refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)
    db.commit()
    release_generation_lock(job.user_id)


def handle_fal_webhook(db: Session, job_id, payload: dict):
    job = db.query(GenerationJob).filter(GenerationJob.id == job_id).with_for_update().first()
    if not job:
        return  # unknown job, ignore safely

    if job.status in ("completed", "failed"):
        return  # already processed — idempotency guard, same pattern as every webhook tonight

    release_generation_lock(job.user_id)

    if payload.get("status") == "OK":
        images = payload.get("payload", {}).get("images", [])
        fal_url = images[0]["url"] if images else None

        if fal_url:
            try:
                file_bytes = download_from_url(fal_url)
            except Exception as e:
                job.status = "failed"
                job.error_message = f"Download from fal failed: {_redact_secrets(e)}"
                refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)
            else:
                try:
                    permanent_path = f"{job.team_id}/{job.feature_type}/{job.id}/output.png"
                    job.output_url = upload_to_storage(permanent_path, file_bytes)
                    job.status = "completed"
                except Exception as e:
                    job.status = "failed"
                    job.error_message = f"Storage upload failed: {_redact_secrets(e)}"
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
    db.commit()