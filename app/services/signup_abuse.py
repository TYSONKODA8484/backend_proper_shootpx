import hashlib
import logging
from datetime import date

from app.core.cache import redis_client

logger = logging.getLogger(__name__)

MAX_BONUS_PER_WINDOW = 5
# 26h, not 24h -- same reasoning as generation_lock's buffer: a fixed
# midnight-keyed date bucket (see _date_bucket) plus a straight 24h TTL would
# let the key expire slightly before the next day's bucket naturally takes
# over under clock/scheduling jitter, briefly exposing a window with no cap
# at all. The extra 2h margin costs nothing (a new date bucket is a new key
# regardless) and only matters at the boundary.
COUNTER_TTL_SECONDS = 26 * 60 * 60

# Small, static list -- not an exhaustive anti-fraud service, just enough to
# stop the laziest bulk-signup scripts from farming free credits with
# throwaway addresses. Skips the BONUS only; the account itself is still
# created normally either way.
DISPOSABLE_EMAIL_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "10minutemail.com", "tempmail.com",
    "throwawaymail.com", "yopmail.com", "trashmail.com", "getnada.com",
    "sharklasers.com", "dispostable.com", "fakeinbox.com", "maildrop.cc",
}


def _date_bucket() -> str:
    return date.today().isoformat()


def _ip_key(ip: str) -> str:
    ip_hash = hashlib.sha256(ip.encode()).hexdigest()
    return f"signup_bonus:ip:{ip_hash}:{_date_bucket()}"


def _device_key(device_id: str) -> str:
    return f"signup_bonus:device:{device_id}:{_date_bucket()}"


def is_disposable_email(email: str | None) -> bool:
    if not email or "@" not in email:
        return False
    domain = email.rsplit("@", 1)[-1].strip().lower()
    return domain in DISPOSABLE_EMAIL_DOMAINS


def _incr_with_ttl(key: str) -> int:
    """INCR, then EXPIRE only on the FIRST increment (mirrors the
    genlock/inflight-count style already used elsewhere) -- an unconditional
    EXPIRE on every call would keep pushing the window out and never let a
    hot key naturally reset."""
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, COUNTER_TTL_SECONDS)
    return count


def should_grant_signup_bonus(ip: str, device_id: str | None) -> bool:
    """
    Returns True if this signup is still under the 5-per-24h cap on BOTH the
    IP and device_id counters and the bonus should be granted -- False if
    either is already at/over the cap. Two Redis round-trips (GET, GET),
    no DB query.
    """
    ip_key = _ip_key(ip)
    device_key = _device_key(device_id) if device_id else None

    ip_count = int(redis_client.get(ip_key) or 0)
    device_count = int(redis_client.get(device_key) or 0) if device_key else 0

    return ip_count < MAX_BONUS_PER_WINDOW and device_count < MAX_BONUS_PER_WINDOW


def record_signup_bonus_grant(ip: str, device_id: str | None) -> None:
    """Call ONLY when the bonus was actually granted -- increments both
    counters so the next signup from this IP/device sees the updated count."""
    _incr_with_ttl(_ip_key(ip))
    if device_id:
        _incr_with_ttl(_device_key(device_id))


def log_signup_bonus_attempt(db, user_id, ip: str, device_id: str | None, bonus_granted: bool) -> None:
    """
    Best-effort audit row -- called via BackgroundTasks (see deps.py) so it
    runs AFTER the response is sent and can never add latency to or fail the
    signup itself. Any failure here is logged, never raised.
    """
    from app.models.signup_bonus_log import SignupBonusLog

    try:
        ip_hash = hashlib.sha256(ip.encode()).hexdigest()
        db.add(SignupBonusLog(
            user_id=user_id, ip_hash=ip_hash, device_id=device_id, bonus_granted=bonus_granted,
        ))
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("failed to write signup_bonus_log for user %s", user_id, exc_info=True)
