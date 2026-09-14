import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


def submit_to_fal(model_id: str, input_params: dict, webhook_url: str) -> str:
    response = httpx.post(
        f"https://queue.fal.run/{model_id}",
        params={"fal_webhook": webhook_url},
        headers={"Authorization": f"Key {settings.fal_key}"},
        json=input_params,
        timeout=30,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        # The exception's own str() only has the status code -- fal's response
        # body is where the actual reason (bad param, rejected image, etc.)
        # lives. Server logs only; job.error_message stays generic.
        logger.exception("fal.ai submit rejected [%s]: %s", response.status_code, response.text)
        raise
    return response.json()["request_id"]

def call_fal_sync(model_id: str, input_params: dict) -> dict:
    """
    For fast, synchronous calls (like a short vision-model query) where we
    can just wait for the answer directly -- no queue, no webhook needed.
    Only use this for calls that genuinely finish in a few seconds; anything
    slower (like the real generation) must use the queue + webhook pattern.
    """
    response = httpx.post(
        f"https://fal.run/{model_id}",
        headers={"Authorization": f"Key {settings.fal_key}"},
        json=input_params,
        timeout=30,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception("fal.ai sync call rejected [%s]: %s", response.status_code, response.text)
        raise
    return response.json()

def upload_image_to_fal(file_bytes: bytes, filename: str, content_type: str) -> str:
    """
    Uploads a local file to fal's CDN, returns the permanent-enough URL to
    use as an input in any subsequent fal model call. Two real HTTP calls:
    initiate (gets a presigned upload URL), then a plain PUT of the bytes.
    """
    initiate_response = httpx.post(
        "https://rest.fal.ai/storage/upload/initiate",
        headers={"Authorization": f"Key {settings.fal_key}"},
        json={"content_type": content_type, "file_name": filename},
        timeout=15,
    )
    initiate_response.raise_for_status()
    data = initiate_response.json()

    upload_url = data["upload_url"]
    file_url = data["file_url"]

    # Presigned URL — no auth header needed here, per fal's own docs
    put_response = httpx.put(upload_url, content=file_bytes, timeout=30)
    put_response.raise_for_status()

    return file_url