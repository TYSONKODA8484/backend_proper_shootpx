"""The shared retry policy (app/core/http_retry.py) and every call site that uses it.

Found live: "fal submit failed: The read operation timed out" failed recolor
jobs outright. The same single-stall exposure existed on the vision/planning
calls and on the image download/upload, so all four now share one policy:
retry timeouts/connection errors/429/502/503/504, never other 4xx, 3 attempts,
1s then 3s backoff.
"""
import httpx
import pytest
from unittest.mock import MagicMock

from app.core import fal_client, http_retry, storage


# ------------------------------- test scaffolding ---------------------------- #

def _response(status=200, body=None):
    request = httpx.Request("POST", "https://example.test/x")
    if body is None and status == 200:
        body = {"request_id": "req_123"}
    return httpx.Response(status, json=body if body is not None else {"detail": "err"}, request=request)


def _binary_response(status=200, content=b"IMGDATA"):
    return httpx.Response(status, content=content, request=httpx.Request("GET", "https://example.test/x"))


class Script:
    """Plays back a scripted list of outcomes (responses or exceptions)."""

    def __init__(self, outcomes):
        self.outcomes, self.calls, self.kwargs = list(outcomes), 0, []

    def __call__(self, *a, **k):
        self.calls += 1
        self.kwargs.append(k)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def slept(monkeypatch):
    sleeps = []
    monkeypatch.setattr(http_retry, "time", MagicMock(sleep=sleeps.append))
    return sleeps


TRANSIENT_ERRORS = [
    httpx.ReadTimeout("The read operation timed out"),
    httpx.ConnectTimeout("connect timed out"),
    httpx.ConnectError("connection refused"),
    httpx.RemoteProtocolError("server disconnected"),
]


# ------------------------------- the policy itself --------------------------- #

def test_policy_constants_match_the_agreed_rules():
    assert http_retry.MAX_ATTEMPTS == 3
    assert http_retry.RETRY_DELAYS_SECONDS == (1.0, 3.0)
    assert http_retry.RETRYABLE_STATUS_CODES == {429, 502, 503, 504}


@pytest.mark.parametrize("error", TRANSIENT_ERRORS, ids=lambda e: type(e).__name__)
def test_transient_transport_errors_are_retried(slept, error):
    script = Script([error, _response()])
    assert http_retry.send_with_retry(script, "t").status_code == 200
    assert script.calls == 2 and slept == [1.0]


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_retryable_statuses_are_retried(slept, status):
    script = Script([_response(status), _response()])
    assert http_retry.send_with_retry(script, "t").status_code == 200
    assert script.calls == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_other_client_errors_are_never_retried(slept, status):
    script = Script([_response(status)])
    assert http_retry.send_with_retry(script, "t").status_code == status
    assert script.calls == 1 and slept == []


def test_backoff_is_one_second_then_three(slept):
    script = Script([httpx.ReadTimeout("t"), httpx.ReadTimeout("t"), _response()])
    assert http_retry.send_with_retry(script, "t").status_code == 200
    assert script.calls == 3 and slept == [1.0, 3.0]


def test_a_persistent_transport_error_raises_after_exactly_three_attempts(slept):
    script = Script([httpx.ReadTimeout("t")] * 3)
    with pytest.raises(httpx.ReadTimeout):
        http_retry.send_with_retry(script, "t")
    assert script.calls == 3 and slept == [1.0, 3.0]   # no sleep after the final failure


def test_a_persistent_503_hands_the_last_response_back_for_normal_error_handling(slept):
    script = Script([_response(503)] * 3)
    assert http_retry.send_with_retry(script, "t").status_code == 503
    assert script.calls == 3


def test_a_first_try_success_never_sleeps(slept):
    script = Script([_response()])
    http_retry.send_with_retry(script, "t")
    assert script.calls == 1 and slept == []


# --------------------- every call site actually uses it ---------------------- #
# One test per site, per behaviour that matters: a transient failure is
# absorbed, a client error is not retried, a persistent failure surfaces.

def _submit():
    return fal_client.submit_to_fal("fal-ai/flux-2/edit", {"prompt": "x"}, "https://backend.test/hook")


def _sync_call():
    return fal_client.call_fal_sync("openrouter/router/vision", {"prompt": "plan shots"})


def _upload():
    return storage.upload_to_storage("t/f/j/out.png", b"IMG")


def _download():
    return storage.download_from_url("https://fal.media/x.png")


# name, callable, httpx verb it uses, builds a success response, checks the result
SITES = [
    ("submit_to_fal", _submit, "post", lambda: _response(), lambda r: r == "req_123"),
    ("call_fal_sync", _sync_call, "post", lambda: _response(body={"output": "ok"}), lambda r: r == {"output": "ok"}),
    ("upload_to_storage", _upload, "post", lambda: _response(body={"Key": "k"}), lambda r: r.endswith("t/f/j/out.png")),
    ("download_from_url", _download, "get", lambda: _binary_response(), lambda r: r == b"IMGDATA"),
]
SITE_IDS = [s[0] for s in SITES]


@pytest.mark.parametrize("name,call,verb,ok,check", SITES, ids=SITE_IDS)
@pytest.mark.parametrize("error", TRANSIENT_ERRORS, ids=lambda e: type(e).__name__)
def test_every_site_absorbs_a_transient_failure(monkeypatch, slept, name, call, verb, ok, check, error):
    script = Script([error, ok()])
    monkeypatch.setattr(httpx, verb, script)
    assert check(call())
    assert script.calls == 2


@pytest.mark.parametrize("name,call,verb,ok,check", SITES, ids=SITE_IDS)
@pytest.mark.parametrize("status", [502, 503, 504, 429])
def test_every_site_retries_gateway_and_rate_limit_responses(monkeypatch, slept, name, call, verb, ok, check, status):
    script = Script([_response(status), ok()])
    monkeypatch.setattr(httpx, verb, script)
    assert check(call())
    assert script.calls == 2


@pytest.mark.parametrize("name,call,verb,ok,check", SITES, ids=SITE_IDS)
@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_no_site_retries_a_client_error(monkeypatch, slept, name, call, verb, ok, check, status):
    script = Script([_response(status)])
    monkeypatch.setattr(httpx, verb, script)
    with pytest.raises(httpx.HTTPStatusError):
        call()
    assert script.calls == 1 and slept == []


@pytest.mark.parametrize("name,call,verb,ok,check", SITES, ids=SITE_IDS)
def test_every_site_fails_after_three_attempts_when_the_outage_persists(monkeypatch, slept, name, call, verb, ok, check):
    script = Script([httpx.ReadTimeout("stalled")] * 3)
    monkeypatch.setattr(httpx, verb, script)
    with pytest.raises(httpx.ReadTimeout):
        call()
    assert script.calls == 3


# ----------------------------- site-specific rules --------------------------- #

def test_storage_upload_sends_x_upsert_on_every_attempt(monkeypatch, slept):
    """Verified against real Supabase: WITHOUT x-upsert a re-upload of the same
    path returns 400 'Duplicate'. A retry after a lost response would then fail
    a job whose image was already safely stored. With it the retry is an
    idempotent overwrite of this job's own file."""
    script = Script([httpx.ReadTimeout("lost the response"), _response(body={"Key": "k"})])
    monkeypatch.setattr(httpx, "post", script)

    storage.upload_to_storage("team/recolor/job/output.png", b"IMG")

    assert script.calls == 2
    for kwargs in script.kwargs:
        assert kwargs["headers"]["x-upsert"] == "true"
        assert kwargs["headers"]["Content-Type"] == "image/png"


def test_storage_upload_still_validates_the_path_before_any_network_call(monkeypatch, slept):
    script = Script([])
    monkeypatch.setattr(httpx, "post", script)
    with pytest.raises(ValueError):
        storage.upload_to_storage("../etc/passwd", b"x")
    assert script.calls == 0


def test_an_expired_download_link_is_not_retried(monkeypatch, slept):
    """fal's output links expire. 403/404 means gone, not 'try again'."""
    script = Script([_binary_response(404, b"gone")])
    monkeypatch.setattr(httpx, "get", script)
    with pytest.raises(httpx.HTTPStatusError):
        storage.download_from_url("https://fal.media/expired.png")
    assert script.calls == 1
