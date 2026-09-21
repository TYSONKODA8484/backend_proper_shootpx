import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from app.core.cache import redis_client
from app.core.config import settings

router = APIRouter()

# arq's own keys. The sentinel is refreshed by the worker's MAIN LOOP (see
# WorkerSettings.health_check_interval), so it vanishes when that loop stops.
WORKER_SENTINEL_KEY = "arq:queue:health-check"
WORKER_QUEUE_KEY = "arq:queue"
# A queued job normally starts within a second or two. Past this, jobs are due
# but nothing is picking them up -- the exact symptom of a frozen worker.
MAX_OVERDUE_SECONDS = 120


@router.get("/")
def root():
    return {"service": settings.app_name, "status": "ok"}


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.get("/health/worker")
def worker_health():
    """
    Is the background worker actually consuming jobs? Meant for an EXTERNAL
    monitor (it keeps working when the worker process is dead or hung, which
    the in-process watchdog cannot cover). 200 when healthy, 503 when not, so
    any uptime tool can alert on it with no custom parsing.

    Two independent signals, either of which fails the check:
      * the arq health sentinel is missing -> the worker's loop hasn't run
        for over ~a minute (or the worker isn't running at all);
      * the oldest due job is overdue by more than MAX_OVERDUE_SECONDS ->
        work is waiting and nothing is taking it. (Precisely tonight's
        symptom: four jobs 204s overdue, never picked up.)
    Public and unauthenticated like /health; it reveals only counts and ages.
    """
    try:
        sentinel = redis_client.get(WORKER_SENTINEL_KEY)
        queue_depth = redis_client.zcard(WORKER_QUEUE_KEY)
        oldest = redis_client.zrange(WORKER_QUEUE_KEY, 0, 0, withscores=True)
    except RedisError:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reasons": ["redis unreachable"]},
        )

    overdue_seconds = 0.0
    if oldest:
        due_at_ms = oldest[0][1]
        overdue_seconds = max(0.0, (time.time() * 1000 - due_at_ms) / 1000)

    reasons = []
    if not sentinel:
        reasons.append("worker heartbeat missing")
    if overdue_seconds > MAX_OVERDUE_SECONDS:
        reasons.append(f"queued work overdue by {overdue_seconds:.0f}s")

    body = {
        "status": "unhealthy" if reasons else "healthy",
        "heartbeat": bool(sentinel),
        "queueDepth": queue_depth,
        "oldestOverdueSeconds": round(overdue_seconds, 1),
    }
    if reasons:
        body["reasons"] = reasons
    return JSONResponse(status_code=503 if reasons else 200, content=body)
