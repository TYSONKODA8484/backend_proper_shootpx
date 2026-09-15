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

def _queue_app_id(model_id: str) -> str:
    """
    submit_to_fal() must use the full model_id, including any trailing
    endpoint-variant segment (e.g. "openai/gpt-image-2/edit" -- "edit" is a
    specific endpoint under the "openai/gpt-image-2" app). But fal's queue
    status/result/cancel endpoints live under the base app id ONLY --
    confirmed live: GET .../openai/gpt-image-2/edit/requests/{id}/status
    returns 405 (Allow: POST) from fal's real API, while GET
    .../openai/gpt-image-2/requests/{id}/status (base app id, no "/edit")
    returns the real status correctly. fal's own response bodies confirm
    this: response_url/status_url on a submit response for this model both
    come back rooted at "openai/gpt-image-2", never including "/edit".
    Stripping to the first two "/"-separated segments matches fal's
    documented owner/app-name convention generally, not just this one model.
    """
    parts = model_id.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else model_id


def check_fal_status(model_id: str, request_id: str) -> dict:
    """
    Real status of an already-submitted request, via fal's queue status
    endpoint -- "status" is one of IN_QUEUE / IN_PROGRESS / COMPLETED. Used
    by the per-tool timeout check to confirm what fal itself actually knows
    before giving up on a job locally purely because our own timeout budget
    elapsed (our budget elapsing does not mean fal's did too).
    """
    response = httpx.get(
        f"https://queue.fal.run/{_queue_app_id(model_id)}/requests/{request_id}/status",
        headers={"Authorization": f"Key {settings.fal_key}"},
        timeout=15,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception("fal.ai status check failed [%s]: %s", response.status_code, response.text)
        raise
    return response.json()


def fetch_fal_result(model_id: str, request_id: str) -> dict:
    """
    Fetches the real output for a request check_fal_status() reported
    COMPLETED for. COMPLETED only means fal finished processing the request
    -- it does not mean the generation itself succeeded, so this call's own
    HTTP status is what actually distinguishes success from failure. Always
    returns fal's own webhook shape ({"status": "OK", "payload": ..., "request_id": ...}
    or {"status": "ERROR", "error": ..., "request_id": ...}) -- including
    "request_id" matters: handle_fal_webhook() checks it against the job's
    own stored fal_request_id before applying anything, so callers can feed
    the result straight into it exactly like a real webhook delivery would.
    """
    response = httpx.get(
        f"https://queue.fal.run/{_queue_app_id(model_id)}/requests/{request_id}",
        headers={"Authorization": f"Key {settings.fal_key}"},
        timeout=15,
    )
    if response.status_code >= 400:
        logger.warning(
            "fal.ai request %s COMPLETED but the result fetch itself failed [%s]: %s",
            request_id, response.status_code, response.text,
        )
        return {"status": "ERROR", "error": f"fal.ai generation failed [{response.status_code}]", "request_id": request_id}

    return {"status": "OK", "payload": response.json(), "request_id": request_id}


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