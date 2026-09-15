"""app/core/fal_client.py -- confirmed live against fal's real API: the queue
status/result/cancel endpoints for a model whose id has more than two
"/"-separated segments (e.g. "openai/gpt-image-2/edit") must be called
against the BASE app id ("openai/gpt-image-2"), not the full submit model_id.
GET on the full id 405s (Allow: POST) and fal's own submit response bodies
confirm response_url/status_url are always rooted at the base app id.

Found live: this silently discarded every real fal output for any
creative_photoshoot/recolor job that ran past its tool's generation_timeout_seconds
locally -- check_fal_status always raised, so check_generation_timeouts always
fell through to failing+refunding the job even when fal had already
succeeded, with the real image sitting unrecovered on fal's side.
"""

from unittest.mock import MagicMock

import httpx
import pytest

from app.core import fal_client


@pytest.mark.parametrize("model_id,expected_base", [
    ("openai/gpt-image-2/edit", "openai/gpt-image-2"),
    ("openai/gpt-image-2.5/sunburst/edit", "openai/gpt-image-2.5"),
    ("openrouter/router", "openrouter/router"),          # already 2 segments -- unchanged
    ("fal-ai/moondream-next", "fal-ai/moondream-next"),  # already 2 segments -- unchanged
])
def test_queue_app_id_strips_any_trailing_endpoint_segment(model_id, expected_base):
    assert fal_client._queue_app_id(model_id) == expected_base


def test_check_fal_status_calls_the_base_app_id_not_the_full_submit_model_id(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        return MagicMock(status_code=200, json=lambda: {"status": "COMPLETED"})

    monkeypatch.setattr(httpx, "get", fake_get)

    fal_client.check_fal_status("openai/gpt-image-2/edit", "req-123")

    assert captured["url"] == "https://queue.fal.run/openai/gpt-image-2/requests/req-123/status"


def test_fetch_fal_result_calls_the_base_app_id_not_the_full_submit_model_id(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        return MagicMock(status_code=200, json=lambda: {"images": [{"url": "https://fal.test/out.png"}]})

    monkeypatch.setattr(httpx, "get", fake_get)

    fal_client.fetch_fal_result("openai/gpt-image-2/edit", "req-123")

    assert captured["url"] == "https://queue.fal.run/openai/gpt-image-2/requests/req-123"


def test_submit_to_fal_still_uses_the_full_model_id_including_the_endpoint_segment(monkeypatch):
    """submit_to_fal must NOT be affected by this fix -- "/edit" is the real,
    required endpoint for the actual generation call, only the queue
    status/result/cancel endpoints need the base app id."""
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        return MagicMock(status_code=200, json=lambda: {"request_id": "req-123"})

    monkeypatch.setattr(httpx, "post", fake_post)

    fal_client.submit_to_fal("openai/gpt-image-2/edit", {"prompt": "x"}, "https://backend.test/webhooks/fal")

    assert captured["url"] == "https://queue.fal.run/openai/gpt-image-2/edit"
