"""Frozen-worker monitoring: the in-process watchdog (restart), the external
GET /health/worker endpoint (alerting), and arq's own health sentinel config.

Background: a worker whose event loop was blocked inside a hung psycopg2 call
sat alive-but-frozen for 40+ minutes -- nothing restarted it (a process manager
only restarts a process that EXITS) and nothing alerted.
"""
import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app import worker
from app.core import watchdog as watchdog_mod
from app.core.watchdog import WorkerWatchdog
from app.main import app
from app.routes import health as health_route


# --------------------------------------------------------------------------- #
# WorkerWatchdog -- decision logic (fake clock, fully deterministic)
# --------------------------------------------------------------------------- #

class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_watchdog_stays_quiet_while_the_loop_is_beating():
    clock, fired = FakeClock(), []
    wd = WorkerWatchdog(180, on_freeze=fired.append, clock=clock)

    clock.now += 170
    assert wd.check_once() is False
    wd.beat()
    clock.now += 170
    assert wd.check_once() is False  # beat reset the window
    assert fired == []


def test_watchdog_fires_once_the_loop_is_silent_past_the_threshold():
    clock, fired = FakeClock(), []
    wd = WorkerWatchdog(180, on_freeze=fired.append, clock=clock)

    clock.now += 180
    assert wd.check_once() is False   # exactly at the threshold is still ok
    clock.now += 1
    assert wd.check_once() is True
    assert fired == [pytest.approx(181)]


# --------------------------------------------------------------------------- #
# The real failure mode: a BLOCKED event loop, watched from another thread
# --------------------------------------------------------------------------- #

def _run_loop_in_thread():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def runner():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    assert ready.wait(2)
    return loop, thread


def test_a_blocked_event_loop_trips_the_watchdog_from_its_own_thread():
    """Reproduces tonight's incident in miniature: a synchronous call that
    never returns (here, a sleep) inside the loop. The loop can't beat, but
    the watchdog THREAD keeps running and fires -- the property the whole
    design rests on."""
    loop, thread = _run_loop_in_thread()
    fired = threading.Event()
    wd = WorkerWatchdog(0.4, on_freeze=lambda age: fired.set(), check_interval=0.05)
    wd.attach_to_loop(loop, interval=0.05)
    wd.start()
    try:
        time.sleep(0.3)
        assert not fired.is_set(), "a healthy, idle loop must not trip the watchdog"

        loop.call_soon_threadsafe(time.sleep, 1.5)   # the hung blocking call
        assert fired.wait(2.5), "watchdog never fired for a blocked event loop"
    finally:
        wd.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(3)


def test_a_healthy_busy_loop_never_trips_the_watchdog():
    loop, thread = _run_loop_in_thread()
    fired = threading.Event()
    wd = WorkerWatchdog(0.4, on_freeze=lambda age: fired.set(), check_interval=0.05)
    wd.attach_to_loop(loop, interval=0.05)
    wd.start()
    try:
        # plenty of short, well-behaved callbacks -- work that yields to the loop
        for _ in range(20):
            loop.call_soon_threadsafe(time.sleep, 0.02)
            time.sleep(0.05)
        assert not fired.is_set()
    finally:
        wd.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(3)


# --------------------------------------------------------------------------- #
# What "fire" actually does: log, alert, hard-exit
# --------------------------------------------------------------------------- #

def test_exit_frozen_worker_logs_critical_alerts_and_hard_exits(monkeypatch, caplog):
    exits, alerts = [], []
    monkeypatch.setattr(watchdog_mod.os, "_exit", exits.append)
    monkeypatch.setattr(watchdog_mod, "send_alert", alerts.append)

    with caplog.at_level("CRITICAL", logger="app.core.watchdog"):
        watchdog_mod.exit_frozen_worker(215.0)

    assert exits == [watchdog_mod.FROZEN_EXIT_CODE] == [70]
    assert "WORKER_FROZEN" in caplog.text          # stable token for log monitors
    assert "215s" in caplog.text
    assert len(alerts) == 1 and "WORKER_FROZEN" in alerts[0]


def test_alert_failure_can_never_prevent_the_restart(monkeypatch):
    """The exit is the whole point. A dead/slow webhook must not stop it."""
    exits = []
    monkeypatch.setattr(watchdog_mod.os, "_exit", exits.append)
    monkeypatch.setattr(watchdog_mod.settings, "alert_webhook_url", "https://hooks.test/x")
    monkeypatch.setattr(watchdog_mod.httpx, "post", MagicMock(side_effect=RuntimeError("webhook down")))

    watchdog_mod.exit_frozen_worker(300.0)

    assert exits == [70]


def test_send_alert_posts_slack_and_discord_compatible_payload(monkeypatch):
    post = MagicMock()
    monkeypatch.setattr(watchdog_mod.settings, "alert_webhook_url", "https://hooks.test/x")
    monkeypatch.setattr(watchdog_mod.httpx, "post", post)

    assert watchdog_mod.send_alert("boom") is True

    post.assert_called_once()
    args, kwargs = post.call_args
    assert args[0] == "https://hooks.test/x"
    assert kwargs["json"] == {"text": "boom", "content": "boom"}
    assert kwargs["timeout"] == watchdog_mod.ALERT_TIMEOUT_SECONDS  # bounded


@pytest.mark.parametrize("url", [None, ""])
def test_send_alert_is_a_noop_without_a_webhook(monkeypatch, url):
    post = MagicMock()
    monkeypatch.setattr(watchdog_mod.settings, "alert_webhook_url", url)
    monkeypatch.setattr(watchdog_mod.httpx, "post", post)

    assert watchdog_mod.send_alert("boom") is False
    post.assert_not_called()


# --------------------------------------------------------------------------- #
# Wired into the real worker
# --------------------------------------------------------------------------- #

def test_worker_startup_arms_the_watchdog_and_shutdown_disarms_it(monkeypatch):
    monkeypatch.setattr(worker.settings, "worker_watchdog_seconds", 999)
    ctx = {}

    async def scenario():
        await worker.startup(ctx)
        assert isinstance(ctx["watchdog"], WorkerWatchdog)
        assert ctx["watchdog"].threshold_seconds == 999
        assert ctx["watchdog"]._thread.is_alive()
        await worker.shutdown(ctx)

    asyncio.run(scenario())
    ctx["watchdog"]._thread.join(2)
    assert not ctx["watchdog"]._thread.is_alive()


def test_watchdog_can_be_disabled_with_zero(monkeypatch):
    monkeypatch.setattr(worker.settings, "worker_watchdog_seconds", 0)
    ctx = {}

    asyncio.run(worker.startup(ctx))

    assert "watchdog" not in ctx


def test_arq_health_sentinel_interval_is_short_enough_to_reveal_a_freeze():
    """arq's default is 3600s -- the sentinel key would outlive a freeze by up
    to an hour and `arq --check` / GET /health/worker would keep reporting
    healthy. Regression guard on that setting."""
    assert worker.WorkerSettings.health_check_interval == 60


# --------------------------------------------------------------------------- #
# GET /health/worker
# --------------------------------------------------------------------------- #

class FakeRedis:
    def __init__(self, sentinel=b"j_complete=1", queue=None, error=None):
        self.sentinel, self.queue, self.error = sentinel, queue or [], error

    def get(self, key):
        if self.error:
            raise self.error
        return self.sentinel

    def zcard(self, key):
        return len(self.queue)

    def zrange(self, key, start, stop, withscores=False):
        return sorted(self.queue, key=lambda x: x[1])[:1]


def _health(monkeypatch, fake):
    monkeypatch.setattr(health_route, "redis_client", fake)
    return TestClient(app).get("/health/worker")


def _due_ms(seconds_ago):
    return time.time() * 1000 - seconds_ago * 1000


def test_worker_health_ok_with_heartbeat_and_an_empty_queue(monkeypatch):
    res = _health(monkeypatch, FakeRedis())
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["heartbeat"] is True


def test_worker_health_ok_when_work_is_queued_but_being_picked_up_promptly(monkeypatch):
    res = _health(monkeypatch, FakeRedis(queue=[("job1", _due_ms(3))]))
    assert res.status_code == 200
    assert res.json()["queueDepth"] == 1


def test_worker_health_503_when_the_heartbeat_is_missing(monkeypatch):
    res = _health(monkeypatch, FakeRedis(sentinel=None))
    assert res.status_code == 503
    assert "worker heartbeat missing" in res.json()["reasons"]


def test_worker_health_503_when_due_jobs_are_not_being_picked_up(monkeypatch):
    """Tonight's exact symptom: four jobs 204s overdue, never picked up."""
    res = _health(monkeypatch, FakeRedis(queue=[("job1", _due_ms(204)), ("job2", _due_ms(203))]))
    assert res.status_code == 503
    body = res.json()
    assert body["queueDepth"] == 2
    assert body["oldestOverdueSeconds"] >= 200
    assert any("overdue" in r for r in body["reasons"])


def test_worker_health_ignores_jobs_scheduled_in_the_future(monkeypatch):
    """A cron entry due in 30s is not 'overdue' -- it must not read as negative
    overdue or trip the check."""
    res = _health(monkeypatch, FakeRedis(queue=[("cron:x", _due_ms(-30))]))
    assert res.status_code == 200
    assert res.json()["oldestOverdueSeconds"] == 0


def test_worker_health_503_when_redis_itself_is_unreachable(monkeypatch):
    res = _health(monkeypatch, FakeRedis(error=RedisConnectionError("down")))
    assert res.status_code == 503
    assert res.json()["reasons"] == ["redis unreachable"]


def test_plain_health_and_root_are_unchanged():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/").json()["status"] == "ok"
