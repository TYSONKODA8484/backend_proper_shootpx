"""get_arq_pool(): lazy singleton with a lock so concurrent first-callers
(e.g. two /generate requests racing at cold start) can't each see _pool as
None and each create a separate pool, leaking whichever connection loses the
race to be the one actually kept.
"""

import asyncio

import pytest

from app.core import arq_pool


@pytest.fixture(autouse=True)
def _reset_pool_singleton():
    """The real module-level singleton must not leak state between tests."""
    arq_pool._pool = None
    yield
    arq_pool._pool = None


def test_get_arq_pool_returns_the_same_pool_on_repeat_calls(monkeypatch):
    created = []

    async def fake_create_pool(settings):
        created.append(settings)
        return object()

    monkeypatch.setattr(arq_pool, "create_pool", fake_create_pool)

    pool_1 = asyncio.run(arq_pool.get_arq_pool())
    pool_2 = asyncio.run(arq_pool.get_arq_pool())

    assert pool_1 is pool_2
    assert len(created) == 1


def test_concurrent_first_calls_only_create_the_pool_once(monkeypatch):
    """The actual race this fixes: two callers both see _pool is None before
    either finishes creating one. Without the lock, both would call
    create_pool(), and whichever finishes first gets silently overwritten
    and leaked."""
    create_calls = []

    async def slow_create_pool(settings):
        create_calls.append(settings)
        await asyncio.sleep(0.05)  # simulate a real connection taking a moment
        return object()

    monkeypatch.setattr(arq_pool, "create_pool", slow_create_pool)

    async def run_concurrently():
        return await asyncio.gather(
            arq_pool.get_arq_pool(),
            arq_pool.get_arq_pool(),
            arq_pool.get_arq_pool(),
        )

    pools = asyncio.run(run_concurrently())

    assert len(create_calls) == 1            # only ever created once
    assert pools[0] is pools[1] is pools[2]  # every caller got the same pool
