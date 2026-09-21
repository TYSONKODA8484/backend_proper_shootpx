"""
Frozen-worker detection for the arq worker.

Why this exists: the arq worker runs every job and every cron on ONE asyncio
event loop, and much of what those jobs do (psycopg2, sync httpx, the sync
redis client) is blocking. If any single blocking call never returns -- a
database connection that died silently mid-query is the real case that took a
worker down for 40+ minutes with nobody noticing -- the loop stops, so nothing
runs, but the OS process stays alive. A process manager only restarts a
process that EXITS, so a frozen-but-alive worker is never restarted, and
because it's alive nothing alerts either.

The fix has to live inside the process, on a thread that keeps running while
the event loop is blocked (blocking C calls release the GIL, so a plain
thread is unaffected):

  * the event loop "beats" (records a timestamp) every few seconds -- but only
    while it is actually free to run scheduled callbacks;
  * a daemon thread checks how long ago the last beat was;
  * past the threshold it alerts, then hard-exits the process
    (os._exit -- a frozen main thread will not honour a graceful shutdown), so
    whatever supervises the worker (Render, systemd Restart=always, Docker
    restart policies, supervisor) brings up a fresh one.
"""
import logging
import os
import threading
import time

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Distinct, greppable exit code (EX_SOFTWARE) so a supervisor's log shows this
# was a deliberate watchdog restart rather than a crash or an OOM kill.
FROZEN_EXIT_CODE = 70
LOOP_BEAT_INTERVAL_SECONDS = 5
CHECK_INTERVAL_SECONDS = 10
ALERT_TIMEOUT_SECONDS = 5


def send_alert(message: str) -> bool:
    """Best-effort POST to ALERT_WEBHOOK_URL (Slack-style {"text"} plus the
    Discord-style {"content"} key, so either kind of incoming webhook accepts
    it). Never raises and is tightly time-bounded: it runs on the way to a
    hard exit, and must not itself become another thing that hangs. Returns
    whether a webhook was actually delivered."""
    url = settings.alert_webhook_url
    if not url:
        return False
    try:
        response = httpx.post(
            url, json={"text": message, "content": message}, timeout=ALERT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return True
    except Exception:
        logger.error("failed to deliver alert webhook", exc_info=True)
        return False


def exit_frozen_worker(age_seconds: float) -> None:
    """The default reaction to a frozen loop: shout, alert, die."""
    message = (
        f"WORKER_FROZEN: the arq worker's event loop has not run for "
        f"{age_seconds:.0f}s (host={os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME') or 'unknown'}, "
        f"pid={os.getpid()}). Queued generations are NOT being processed. "
        f"Exiting so the process manager restarts it."
    )
    # CRITICAL + a stable token so a log-based monitor can match on it.
    logger.critical(message)
    send_alert(message)
    os._exit(FROZEN_EXIT_CODE)


class WorkerWatchdog:
    def __init__(self, threshold_seconds: float, on_freeze=exit_frozen_worker,
                 clock=time.monotonic, check_interval: float = CHECK_INTERVAL_SECONDS):
        self.threshold_seconds = threshold_seconds
        self.on_freeze = on_freeze
        self.clock = clock
        self.check_interval = check_interval
        self._last_beat = clock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self) -> None:
        self._last_beat = self.clock()

    def age(self) -> float:
        return self.clock() - self._last_beat

    def check_once(self) -> bool:
        """True (and fires on_freeze) if the loop has been silent too long."""
        age = self.age()
        if age > self.threshold_seconds:
            self.on_freeze(age)
            return True
        return False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="worker-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        # Event.wait, not sleep, so stop() takes effect immediately.
        while not self._stop.wait(self.check_interval):
            try:
                if self.check_once():
                    return
            except Exception:
                # The watchdog must never die quietly -- it is the thing that
                # notices everything else dying.
                logger.exception("worker watchdog check failed")

    def attach_to_loop(self, loop, interval: float = LOOP_BEAT_INTERVAL_SECONDS) -> None:
        """Beat from inside the event loop itself. call_later callbacks only
        run when the loop is free, so a blocked loop stops beating -- which is
        exactly the signal the watchdog thread is watching for."""
        def _tick():
            self.beat()
            loop.call_later(interval, _tick)

        loop.call_soon(_tick)
