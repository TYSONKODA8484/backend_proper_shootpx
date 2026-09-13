"""fal.ai webhook signature verification.

Scheme is fal's actual current one (verified against
https://fal.ai/docs/model-apis/model-endpoints/webhooks), NOT a generic
ED25519 check invented for this project:

- Required headers: X-Fal-Webhook-Request-Id, X-Fal-Webhook-User-Id,
  X-Fal-Webhook-Timestamp, X-Fal-Webhook-Signature.
- Timestamp must be within +/-300s of now (replay protection).
- Message = request_id + "\n" + user_id + "\n" + timestamp + "\n" +
  sha256_hex(raw_request_body), signed with ED25519, signature sent hex-encoded.
- Public keys are served as a JWKS (https://rest.fal.ai/.well-known/jwks.json),
  each key's "x" field a base64url-encoded raw 32-byte Ed25519 public key.
  Any key in the set validating the signature is a pass.
"""

import base64
import hashlib
import logging
import time

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from app.core.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

JWKS_URL = "https://rest.fal.ai/.well-known/jwks.json"
JWKS_CACHE_KEY = "fal:webhook:jwks"
JWKS_CACHE_TTL_SECONDS = 24 * 3600  # fal's own stated max cache duration
JWKS_FETCH_TIMEOUT_SECONDS = 5
TIMESTAMP_TOLERANCE_SECONDS = 300  # +/-5 min, per fal's docs

REQUIRED_HEADERS = (
    "X-Fal-Webhook-Request-Id",
    "X-Fal-Webhook-User-Id",
    "X-Fal-Webhook-Timestamp",
    "X-Fal-Webhook-Signature",
)


class FalWebhookVerificationError(Exception):
    """Raised for any webhook that fails verification. The caller must reject
    the request without acting on its payload at all."""


def _fetch_jwks() -> list[dict]:
    cached = get_cached(JWKS_CACHE_KEY)
    if cached is not None:
        return cached

    try:
        resp = requests.get(JWKS_URL, timeout=JWKS_FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        keys = resp.json().get("keys", [])
    except Exception:
        logger.warning("failed to fetch fal.ai JWKS", exc_info=True)
        return []

    set_cached(JWKS_CACHE_KEY, keys, ttl=JWKS_CACHE_TTL_SECONDS)
    return keys


def _decode_b64url(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded)


def verify_fal_webhook(headers, raw_body: bytes) -> None:
    """Raises FalWebhookVerificationError on any failure. Only when this
    returns normally has the request been proven to come from fal.ai."""
    values = {}
    for name in REQUIRED_HEADERS:
        value = headers.get(name)
        if not value:
            raise FalWebhookVerificationError(f"Missing required header: {name}")
        values[name] = value

    request_id = values["X-Fal-Webhook-Request-Id"]
    user_id = values["X-Fal-Webhook-User-Id"]
    timestamp = values["X-Fal-Webhook-Timestamp"]
    signature_hex = values["X-Fal-Webhook-Signature"]

    try:
        ts = int(timestamp)
    except ValueError:
        raise FalWebhookVerificationError("Invalid X-Fal-Webhook-Timestamp header")

    if abs(int(time.time()) - ts) > TIMESTAMP_TOLERANCE_SECONDS:
        raise FalWebhookVerificationError("Webhook timestamp outside allowed window")

    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        raise FalWebhookVerificationError("Malformed X-Fal-Webhook-Signature header")

    body_hash = hashlib.sha256(raw_body).hexdigest()
    message = f"{request_id}\n{user_id}\n{timestamp}\n{body_hash}".encode()

    keys = _fetch_jwks()
    if not keys:
        # Fail closed: an unverifiable webhook is never trusted, even if that
        # means a transient JWKS outage rejects genuine deliveries.
        raise FalWebhookVerificationError("No fal.ai verification keys available")

    for key in keys:
        try:
            public_key = Ed25519PublicKey.from_public_bytes(_decode_b64url(key["x"]))
            public_key.verify(signature, message)
            return  # a single matching key is sufficient
        except (InvalidSignature, KeyError, ValueError):
            continue

    raise FalWebhookVerificationError("Signature verification failed")
