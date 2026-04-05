"""Tests for request rate limiting."""

from __future__ import annotations

import time

import pytest

from xpyd_acc.rate_limit import RateLimiter


@pytest.mark.asyncio
async def test_unlimited_no_delay():
    """RateLimiter with None rate should not delay."""
    rl = RateLimiter(None)
    start = time.monotonic()
    for _ in range(10):
        await rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.1
    assert rl.throttle_count == 0


@pytest.mark.asyncio
async def test_zero_rate_unlimited():
    """Rate of 0 or negative should behave as unlimited."""
    for rate in (0, -1.0):
        rl = RateLimiter(rate)
        assert rl.rate is None
        await rl.acquire()
        assert rl.throttle_count == 0


@pytest.mark.asyncio
async def test_rate_property():
    """Rate property returns configured value."""
    rl = RateLimiter(5.0)
    assert rl.rate == 5.0


@pytest.mark.asyncio
async def test_rate_limiter_throttles():
    """With a low rate, acquire should introduce delays."""
    rl = RateLimiter(2.0)  # 2 requests/second
    # First 2 should be fast (bucket starts full)
    await rl.acquire()
    await rl.acquire()
    # Third should be throttled
    start = time.monotonic()
    await rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.3  # Should wait ~0.5s but allow some margin
    assert rl.throttle_count >= 1


@pytest.mark.asyncio
async def test_throttle_count_increments():
    """Throttle count should increment when waiting."""
    rl = RateLimiter(1.0)  # 1 req/s
    await rl.acquire()  # immediate
    assert rl.throttle_count == 0
    await rl.acquire()  # should wait
    assert rl.throttle_count >= 1


def test_env_var_support():
    """XPYD_ACC_RATE_LIMIT env var should be picked up."""
    import os

    os.environ["XPYD_ACC_RATE_LIMIT"] = "5.5"
    try:
        from xpyd_acc.env import get_env_defaults
        env = get_env_defaults()
        assert env.rate_limit == 5.5
    finally:
        del os.environ["XPYD_ACC_RATE_LIMIT"]


def test_config_rate_limit():
    """Config should support rate_limit in defaults."""
    from xpyd_acc.config import DefaultsConfig
    dc = DefaultsConfig(rate_limit=10.0)
    assert dc.rate_limit == 10.0


def test_config_rate_limit_default_none():
    """Default rate_limit should be None."""
    from xpyd_acc.config import DefaultsConfig
    dc = DefaultsConfig()
    assert dc.rate_limit is None
