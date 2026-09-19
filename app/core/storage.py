import logging
import re

import httpx
from app.core.config import settings
from app.core.http_retry import send_with_retry

logger = logging.getLogger(__name__)

BUCKET_NAME = "generation-storage"

# Every real caller composes this path from a team_id/job_id (always plain
# UUIDs) and a feature_type that must already exist as a tool_definitions
# primary key -- but this is the actual boundary that touches external
# storage, so it validates the composed path itself rather than trusting
# that invariant to hold forever upstream.
_SAFE_PATH = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]|/(?!\.\.?(?:/|$)))*$")


def _validate_storage_path(file_path: str) -> None:
    if not file_path or "\\" in file_path or not _SAFE_PATH.match(file_path):
        raise ValueError(f"Unsafe storage path: {file_path!r}")


def upload_to_storage(file_path: str, file_bytes: bytes, content_type: str = "image/png") -> str:
    """
    Uploads a file to the Supabase Storage bucket at the given path.
    Returns the public URL where it can be accessed.
    """
    _validate_storage_path(file_path)
    url = f"{settings.supabase_url}/storage/v1/object/{BUCKET_NAME}/{file_path}"

    # Retried on transient failures (app/core/http_retry.py). A retry after an
    # ambiguous read timeout may land on an upload that in fact succeeded the
    # first time; without x-upsert Supabase would answer that with a 400
    # "Duplicate" (a 4xx, never retried), failing a job whose image is already
    # safely stored. x-upsert makes the retry an idempotent overwrite. The path
    # is unique per job (team/feature/job_id), so upsert can only ever replace
    # this same job's own file.
    response = send_with_retry(
        lambda: httpx.post(
            url,
            # Supabase's newer sb_secret_... keys are not JWTs, so they must go on
            # the `apikey` header -- `Authorization: Bearer` tries to parse them
            # as a JWT and fails ("Invalid Compact JWS"). Confirmed live against
            # this project's real key during this audit.
            headers={
                "apikey": settings.supabase_service_role_key,
                "Content-Type": content_type,
                "x-upsert": "true",
            },
            content=file_bytes,
            timeout=30,
        ),
        "Supabase storage upload",
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        # The exception's own str() only has the status code -- Supabase's
        # response body is where the real reason (bad bucket, RLS rejection,
        # bad key) lives. Caller (handle_fal_webhook) already stores only a
        # generic message on the job; this is what makes the real cause
        # visible in server logs.
        logger.exception("Supabase storage upload rejected [%s]: %s", response.status_code, response.text)
        raise

    return f"{settings.supabase_url}/storage/v1/object/public/{BUCKET_NAME}/{file_path}"


def download_from_url(url: str) -> bytes:
    """
    Downloads a file from any URL (used to fetch fal.ai's output before
    it expires, so we can re-upload it to our own permanent storage).
    """
    # Retried on transient failures (app/core/http_retry.py). A plain GET, so a
    # retry is always safe. A 403/404 (fal's output link expired or was never
    # valid) is a client error and is NOT retried.
    response = send_with_retry(lambda: httpx.get(url, timeout=30), "fal.ai output download")
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception("Failed to download fal.ai output [%s]: %s", response.status_code, response.text)
        raise
    return response.content