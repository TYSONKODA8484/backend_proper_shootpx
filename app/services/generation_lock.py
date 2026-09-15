from app.core.cache import redis_client

LOCK_TTL_SECONDS = 300  # 5 min safety net — auto-expires if a job dies without releasing
INFLIGHT_KEY_GLOBAL = "fal:inflight_count"


def acquire_generation_lock(user_id) -> bool:
    """
    Returns True if the lock was acquired (no generation currently running
    for this user). Returns False if the user already has one in progress.
    """
    key = f"genlock:user:{user_id}"
    return redis_client.set(key, "1", nx=True, ex=LOCK_TTL_SECONDS) is not None


def release_generation_lock(user_id) -> None:
    key = f"genlock:user:{user_id}"
    redis_client.delete(key)


def generation_lock_age_seconds(user_id):
    """
    None if no lock is currently held for this user; otherwise roughly how
    many seconds ago it was acquired (derived from the key's remaining TTL,
    since Redis doesn't track acquisition time directly).

    Used to self-heal a lock orphaned by a crash/restart between
    acquire_generation_lock() and the job actually being created or the
    except-block release running -- a hard process kill mid-request skips
    that cleanup entirely, otherwise stranding the lock for the full 5-minute
    TTL. See create_generation_batch's stale-lock check.
    """
    key = f"genlock:user:{user_id}"
    ttl = redis_client.ttl(key)
    if ttl is None or ttl < 0:
        return None
    return LOCK_TTL_SECONDS - ttl


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