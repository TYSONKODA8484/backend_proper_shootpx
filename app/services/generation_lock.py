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


def try_reserve_fal_slot(team_id) -> bool:
    """
    Two checks: the account-wide fal concurrency limit (never exceed fal's
    real ceiling), AND a per-team cap (so one team's large batch can't
    consume every slot and starve every other team using the product at
    the same time).
    """
    from app.core.config import settings

    global_current = redis_client.incr(INFLIGHT_KEY_GLOBAL)
    if global_current > settings.fal_concurrency_limit:
        redis_client.decr(INFLIGHT_KEY_GLOBAL)
        return False

    team_key = f"fal:inflight_count:{team_id}"
    team_current = redis_client.incr(team_key)
    if team_current > settings.fal_per_team_concurrency_limit:
        redis_client.decr(team_key)
        redis_client.decr(INFLIGHT_KEY_GLOBAL)
        return False

    return True


def release_fal_slot(team_id) -> None:
    redis_client.decr(INFLIGHT_KEY_GLOBAL)
    redis_client.decr(f"fal:inflight_count:{team_id}")