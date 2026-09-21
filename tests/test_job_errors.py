"""public_job_error: the ONLY error text an API response may carry for a job.

Found live: users saw raw internals in the UI -- "fal submit failed: The read
operation timed out", "fal.ai generation failed [422]", "fal reported success
but returned no image". job.error_message is internal; it must never be shown.
"""
import pytest

from app import worker
from app.services import job_errors
from app.services.generation import (
    DOWNLOAD_FAILED_MESSAGE, SWEEP_TIMEOUT_MESSAGE, TIMEOUT_MESSAGE, UPLOAD_FAILED_MESSAGE,
)
from app.services.job_errors import (
    FRIENDLY_TIMEOUT_MESSAGE, GENERIC_FAILURE_MESSAGE, public_job_error,
)

# Every distinct raw message that has actually been written to a job in the
# real database, plus the shapes the code can produce.
REAL_INTERNAL_MESSAGES = [
    "fal submit failed: The read operation timed out",
    "fal submit failed: [Errno 11001] getaddrinfo failed",
    "fal.ai generation failed [422]",
    "Unexpected status code: 422",
    "fal reported success but returned no image",
    "fal reported success but returned no text",
    "Generation failed",
    "Manually cleared: job never reached the worker (stale test job)",
    "Orphaned by a manual worker restart during testing -- cleaned up manually.",
    "some brand new internal message nobody has allowlisted",
    "",
]


@pytest.mark.parametrize("raw", REAL_INTERNAL_MESSAGES)
def test_internal_messages_never_reach_the_user(raw):
    shown = public_job_error("failed", raw)
    assert shown == GENERIC_FAILURE_MESSAGE
    for leaked in ("fal", "Errno", "422", "status code", "timed out", "worker", "test job"):
        assert leaked.lower() not in shown.lower()


def test_a_failed_job_with_no_recorded_error_still_gets_a_message():
    assert public_job_error("failed", None) == GENERIC_FAILURE_MESSAGE


@pytest.mark.parametrize("status", ["queued", "processing", "completed", None])
def test_non_failed_jobs_carry_no_error_at_all(status):
    """Even if a stale raw message is somehow still on the row."""
    assert public_job_error(status, "fal submit failed: leaked") is None


@pytest.mark.parametrize("raw", [TIMEOUT_MESSAGE, SWEEP_TIMEOUT_MESSAGE])
def test_timeouts_become_one_friendly_message(raw):
    assert public_job_error("failed", raw) == FRIENDLY_TIMEOUT_MESSAGE
    assert "no response received" not in FRIENDLY_TIMEOUT_MESSAGE


@pytest.mark.parametrize("raw", [DOWNLOAD_FAILED_MESSAGE, UPLOAD_FAILED_MESSAGE])
def test_messages_already_written_for_users_pass_through_unchanged(raw):
    assert public_job_error("failed", raw) == raw


def test_the_start_failed_message_matches_the_workers_own_constant():
    """The worker can't be imported into the API process, so the allowlist
    holds its own copy -- pinned here so a reword in the worker can't silently
    turn a user-safe message into the generic one (or worse, drift)."""
    assert job_errors.START_FAILED_MESSAGE == worker.GENERIC_START_FAILED_MESSAGE
    assert public_job_error("failed", worker.GENERIC_START_FAILED_MESSAGE) == worker.GENERIC_START_FAILED_MESSAGE


def test_every_allowlisted_output_is_itself_safe():
    """Nothing the allowlist can produce may contain provider/internal words."""
    for shown in set(job_errors._PUBLIC_MESSAGES.values()) | {GENERIC_FAILURE_MESSAGE}:
        for bad in ("fal", "Errno", "status code", "Traceback", "worker", "openrouter", "supabase"):
            assert bad.lower() not in shown.lower(), (bad, shown)
