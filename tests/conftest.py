import pytest

from app.core.limiter import limiter


@pytest.fixture(autouse=True)
def _disable_rate_limiter():
    """Rate limiting uses a shared Redis store; disable it in tests so counters
    from one test (or a previous run) don't make another test flaky."""
    limiter.enabled = False
    yield
    limiter.enabled = True
