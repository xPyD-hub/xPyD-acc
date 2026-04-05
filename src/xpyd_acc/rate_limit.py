"""Request rate limiting with token bucket algorithm."""

from __future__ import annotations

import asyncio
import time

from xpyd_acc.log import get_logger

logger = get_logger("rate_limit")


class RateLimiter:
    """Async token bucket rate limiter.

    Args:
        rate: Maximum requests per second. None or <= 0 means unlimited.
    """

    def __init__(self, rate: float | None = None) -> None:
        self._rate = rate if rate is not None and rate > 0 else None
        if self._rate is not None:
            self._tokens = self._rate
            self._last_refill = time.monotonic()
            self._lock = asyncio.Lock()
        self._throttle_count = 0

    @property
    def rate(self) -> float | None:
        """Configured rate limit (requests/second), or None if unlimited."""
        return self._rate

    @property
    def throttle_count(self) -> int:
        """Number of times a request was throttled (had to wait)."""
        return self._throttle_count

    async def acquire(self) -> None:
        """Wait until a request is allowed under the rate limit."""
        if self._rate is None:
            return

        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self._rate
                self._throttle_count += 1
                logger.info("Rate limited: waiting %.3fs", wait_time)
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
                self._last_refill = time.monotonic()
            else:
                self._tokens -= 1.0
