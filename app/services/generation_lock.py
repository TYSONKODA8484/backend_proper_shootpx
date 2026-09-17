from app.core.cache import redis_client

# Generic safety net used only until the specific tool being generated is
# known (acquire_generation_lock() itself runs before that lookup in most
# callers) -- extend_generation_lock_ttl() below re-derives the REAL TTL from
# that tool's own generation_timeout_seconds once it's available. Any tool
# whose generation_timeout_seconds reaches or exceeds this fixed value would
# otherwise have its lock silently expire while the job is still legitimately
# running within its own timeout budget -- confirmed live: listing_photoshoot
# is configured at exactly 300s, matching this constant precisely.
LOCK_TTL_SECONDS = 300
# Margin added on top of a tool's own generation_timeout_seconds so the lock
# always outlives the job's worst-case runtime (network/queue jitter, the
# worker's own delivery-poll cadence) rather than expiring right at the wire.
LOCK_TTL_BUFFER_SECONDS = 30
INFLIGHT_KEY_GLOBAL = "fal:inflight_count"


def acquire_generation_lock(user_id, ttl_seconds: int = LOCK_TTL_SECONDS) -> bool:
    """
    Returns True if the lock was acquired (no generation currently running
    for this user). Returns False if the user already has one in progress.

    The lock's value is the TTL it was set with (not a placeholder) -- that's
    what lets generation_lock_age_seconds() compute age correctly for a lock
    later extended to a tool-specific TTL via extend_generation_lock_ttl(),
    instead of assuming every lock used the same fixed LOCK_TTL_SECONDS.
    """
    key = f"genlock:user:{user_id}"
    return redis_client.set(key, str(ttl_seconds), nx=True, ex=ttl_seconds) is not None


def extend_generation_lock_ttl(user_id, generation_timeout_seconds: int) -> None:
    """
    Re-derives the lock's TTL from the tool's own generation_timeout_seconds
    (+ LOCK_TTL_BUFFER_SECONDS) once that value is known -- acquire_generation_
    lock() itself is usually called before the specific tool is looked up, so
    it starts with the generic LOCK_TTL_SECONDS safety net above. Called from
    create_generation_batch as soon as the tool row is resolved, before any
    of the (potentially slow) work after it.

    A no-op if the lock isn't currently held (SET ... XX only applies to an
    existing key) -- nothing to extend, and it must never resurrect a lock
    that was already released.
    """
    key = f"genlock:user:{user_id}"
    new_ttl = generation_timeout_seconds + LOCK_TTL_BUFFER_SECONDS
    redis_client.set(key, str(new_ttl), xx=True, ex=new_ttl)


def release_generation_lock(user_id) -> None:
    key = f"genlock:user:{user_id}"
    redis_client.delete(key)


def generation_lock_age_seconds(user_id):
    """
    None if no lock is currently held for this user; otherwise roughly how
    many seconds ago it was acquired (derived from the key's remaining TTL
    against the TTL it was actually set with -- stored as the key's own
    value, since Redis doesn't track acquisition time directly, and a lock's
    real TTL can now differ per tool via extend_generation_lock_ttl()).

    Used to self-heal a lock orphaned by a crash/restart between
    acquire_generation_lock() and the job actually being created or the
    except-block release running -- a hard process kill mid-request skips
    that cleanup entirely, otherwise stranding the lock for its full TTL.
    See create_generation_batch's stale-lock check.
    """
    key = f"genlock:user:{user_id}"
    ttl = redis_client.ttl(key)
    if ttl is None or ttl < 0:
        return None

    raw_original_ttl = redis_client.get(key)
    try:
        original_ttl = int(raw_original_ttl)
    except (TypeError, ValueError):
        # Defensive fallback only -- every lock set by this module's own
        # acquire_generation_lock() stores its real TTL as its value, so this
        # only triggers against a corrupted/foreign value at this key.
        original_ttl = LOCK_TTL_SECONDS

    return original_ttl - ttl


def try_reserve_fal_slot(team_id) -> bool:
    """
    Enforces the per-team cap only: fal has no per-team concept of its own,
    so one team's large batch could otherwise consume every in-flight slot
    and starve every other team using the product at the same time -- that
    fairness enforcement still has to happen on our side.

    There used to be an account-wide cap enforced here too, with a
    retry-and-re-enqueue loop in submit_generation_to_fal for whenever it was
    hit. Removed: fal's own documented behavior already queues and dispatches
    submissions automatically once their account-wide limit is reached, so
    pre-checking and retrying for it on our side was redundant work -- and,
    combined with a corrupted counter, was the source of a real repeated-
    request incident. The global inflight count below is still tracked
    (paired with release_fal_slot) purely as an observability metric now,
    not as a gate.
    """
    from app.core.config import settings

    redis_client.incr(INFLIGHT_KEY_GLOBAL)

    team_key = f"fal:inflight_count:{team_id}"
    try:
        team_current = redis_client.incr(team_key)
    except Exception:
        # The global increment above already landed -- if the team-level one
        # blows up (transient Redis error, or a corrupted non-integer value
        # sitting at this specific key), the global counter must not be left
        # permanently +1 with nothing to ever decrement it back.
        redis_client.decr(INFLIGHT_KEY_GLOBAL)
        raise

    if team_current > settings.fal_per_team_concurrency_limit:
        redis_client.decr(team_key)
        redis_client.decr(INFLIGHT_KEY_GLOBAL)
        return False

    return True


def release_fal_slot(team_id) -> None:
    redis_client.decr(INFLIGHT_KEY_GLOBAL)
    redis_client.decr(f"fal:inflight_count:{team_id}")