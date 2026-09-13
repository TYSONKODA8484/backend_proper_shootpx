import re

import httpx
from app.core.config import settings

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

    response = httpx.post(
        url,
        # Supabase's newer sb_secret_... keys are not JWTs, so they must go on
        # the `apikey` header -- `Authorization: Bearer` tries to parse them
        # as a JWT and fails ("Invalid Compact JWS"). Confirmed live against
        # this project's real key during this audit.
        headers={
            "apikey": settings.supabase_service_role_key,
            "Content-Type": content_type,
        },
        content=file_bytes,
        timeout=30,
    )
    response.raise_for_status()

    return f"{settings.supabase_url}/storage/v1/object/public/{BUCKET_NAME}/{file_path}"


def download_from_url(url: str) -> bytes:
    """
    Downloads a file from any URL (used to fetch fal.ai's output before
    it expires, so we can re-upload it to our own permanent storage).
    """
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    return response.content