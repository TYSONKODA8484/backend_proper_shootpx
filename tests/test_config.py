"""app/core/config.py::Settings -- a secret pasted into Render's env var UI
(or a .env file saved with a trailing newline) can silently carry a trailing
"\\n" into the value. That's invisible everywhere except when later sent as a
raw HTTP header, where h11 rejects it outright
(httpx.LocalProtocolError: "Illegal header value ...\\n").

Found live: SUPABASE_SERVICE_ROLE_KEY had a trailing newline on Render,
breaking every real storage upload with exactly that error -- the request
never even reached Supabase.
"""

import os

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import Settings


def test_settings_strips_whitespace_configured():
    """Pin the actual mechanism: str_strip_whitespace must stay enabled on
    Settings, or this whole class of bug comes back silently."""
    assert Settings.model_config.get("str_strip_whitespace") is True


def test_trailing_newline_in_env_var_is_stripped(monkeypatch):
    """Reproduces the real bug with a minimal Settings-shaped model (the real
    Settings requires every env var in .env.example, which is impractical to
    fully stub here) -- proves str_strip_whitespace actually removes a
    trailing newline the way Render's env var UI can introduce one."""

    class Probe(BaseSettings):
        model_config = SettingsConfigDict(str_strip_whitespace=True)
        supabase_service_role_key: str

    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_ft6xNG6hYmTcZWbv83dU-Q_ofH5If7Z\n")

    probe = Probe()

    assert probe.supabase_service_role_key == "sb_secret_ft6xNG6hYmTcZWbv83dU-Q_ofH5If7Z"
    assert "\n" not in probe.supabase_service_role_key
