from app.core.cache import redis_client

LOCK_TTL_SECONDS = 300  # 5 min safety net — auto-expires if a job dies without releasing


def acquire_generation_lock(user_id) -> bool:
    """
    Returns True if the lock was acquired (no generation currently running
    for this user). Returns False if the user already has one in progress.
    """
    key = f"genlock:user:{user_id}"
    # SET ... NX = only set if the key doesn't already exist (atomic check-and-set)
    return redis_client.set(key, "1", nx=True, ex=LOCK_TTL_SECONDS) is not None


def release_generation_lock(user_id) -> None:
    key = f"genlock:user:{user_id}"
    redis_client.delete(key)