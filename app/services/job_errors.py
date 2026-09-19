"""
What a user is allowed to be told when a generation fails.

job.error_message is an INTERNAL field. It is written from many places and a
lot of what lands in it is provider or exception text that must never reach a
customer: "fal submit failed: The read operation timed out", "fal.ai
generation failed [422]", "Unexpected status code: 422", "fal reported success
but returned no image". Showing those tells the user nothing useful, exposes
which vendor we use and how our pipeline is wired, and makes a brief hiccup
look like a broken product.

So the raw value is kept (in the DB, for support and debugging) and only ever
exposed through public_job_error(), an ALLOWLIST: a message is shown as-is only
if this codebase wrote it specifically for users; everything else -- including
any new internal message added later -- becomes the generic one. Failing closed
means a future developer can't accidentally leak a new internal string just by
writing it into error_message.
"""
from app.services.generation import (
    DOWNLOAD_FAILED_MESSAGE,
    SWEEP_TIMEOUT_MESSAGE,
    TIMEOUT_MESSAGE,
    UPLOAD_FAILED_MESSAGE,
)

GENERIC_FAILURE_MESSAGE = "We couldn't generate this image. Please try again."
FRIENDLY_TIMEOUT_MESSAGE = "This took longer than expected. Please try again."

# Must match app/worker.py's GENERIC_START_FAILED_MESSAGE (pinned by a test --
# the worker can't be imported here without dragging the whole worker into the
# API process).
START_FAILED_MESSAGE = "Failed to start generation. Please try again."

# raw message written by our code  ->  what the user sees
_PUBLIC_MESSAGES = {
    TIMEOUT_MESSAGE: FRIENDLY_TIMEOUT_MESSAGE,
    SWEEP_TIMEOUT_MESSAGE: FRIENDLY_TIMEOUT_MESSAGE,  # "...no response received" is internal-sounding
    DOWNLOAD_FAILED_MESSAGE: DOWNLOAD_FAILED_MESSAGE,  # already written for users
    UPLOAD_FAILED_MESSAGE: UPLOAD_FAILED_MESSAGE,
    START_FAILED_MESSAGE: START_FAILED_MESSAGE,
}


def public_job_error(status: str | None, raw_error: str | None) -> str | None:
    """The only error text an API response may carry for a job."""
    if status != "failed":
        return None  # nothing failed, so there is nothing to explain
    return _PUBLIC_MESSAGES.get(raw_error, GENERIC_FAILURE_MESSAGE)
