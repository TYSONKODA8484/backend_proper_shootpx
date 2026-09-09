from unittest.mock import MagicMock

import app.core.limiter as limiter_mod
from app.core.limiter import client_ip


def _req(xff=None, host="127.0.0.1"):
    r = MagicMock()
    r.headers = {"x-forwarded-for": xff} if xff else {}
    r.client.host = host
    return r


def test_ignores_forwarded_header_when_proxy_not_trusted(monkeypatch):
    monkeypatch.setattr(limiter_mod.settings, "trust_proxy", False)
    assert client_ip(_req(xff="9.9.9.9", host="127.0.0.1")) == "127.0.0.1"


def test_uses_last_forwarded_hop_when_proxy_trusted(monkeypatch):
    monkeypatch.setattr(limiter_mod.settings, "trust_proxy", True)
    # client-spoofed "1.2.3.4" first, real client appended by the trusted proxy
    assert client_ip(_req(xff="1.2.3.4, 203.0.113.7")) == "203.0.113.7"


def test_falls_back_to_socket_ip_when_no_header(monkeypatch):
    monkeypatch.setattr(limiter_mod.settings, "trust_proxy", True)
    assert client_ip(_req(host="198.51.100.2")) == "198.51.100.2"
