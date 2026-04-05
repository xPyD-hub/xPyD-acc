"""Reusable async HTTP retry with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import httpx

from xpyd_acc.log import get_logger

T = TypeVar("T")

logger = get_logger("retry")

#: HTTP status codes that trigger a retry.
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


class RetriesExhausted(Exception):
    """All retry attempts failed."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"All {attempts} attempts failed. Last error: {last_error}")


@dataclass
class RetryResult:
    """Wrapper returned by :func:`retry_async` containing the value and attempt metadata."""

    value: Any
    attempts: int  # 1 means succeeded on first try (no retries)


@dataclass
class RetryStats:
    """Aggregate retry statistics across multiple requests."""

    total_requests: int = 0
    total_retries: int = 0  # sum of (attempts - 1) across all requests
    max_retries_single: int = 0  # highest (attempts - 1) for a single request
    retried_request_count: int = 0  # how many requests needed at least one retry

    def record(self, result: RetryResult) -> None:
        """Record a single :class:`RetryResult`."""
        retries = result.attempts - 1
        self.total_requests += 1
        self.total_retries += retries
        if retries > self.max_retries_single:
            self.max_retries_single = retries
        if retries > 0:
            self.retried_request_count += 1

    def to_dict(self) -> dict[str, int]:
        """Serialize to a plain dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_retries": self.total_retries,
            "max_retries_single": self.max_retries_single,
            "retried_request_count": self.retried_request_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetryStats:
        """Deserialize from a dictionary."""
        return cls(
            total_requests=data.get("total_requests", 0),
            total_retries=data.get("total_retries", 0),
            max_retries_single=data.get("max_retries_single", 0),
            retried_request_count=data.get("retried_request_count", 0),
        )


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> RetryResult:
    """Call *func* with retries on transient HTTP errors.

    Retries on:
    - ``httpx.ConnectError``, ``httpx.TimeoutException``
    - HTTP responses with status in ``RETRYABLE_STATUS_CODES``

    When a 429 response includes a ``Retry-After`` header the wait honours that
    value instead of the computed backoff.

    Args:
        func: Async callable to invoke.
        retries: Maximum number of attempts (total, not extra retries).
        base_delay: Base delay in seconds for exponential backoff.
        *args, **kwargs: Forwarded to *func*.

    Returns:
        A :class:`RetryResult` containing the return value and the number of
        attempts made.

    Raises:
        RetriesExhausted: If all attempts fail.
    """
    if retries < 1:
        retries = 1

    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            value = await func(*args, **kwargs)
            return RetryResult(value=value, attempts=attempt + 1)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                raise
            last_error = exc
            delay = _compute_delay(exc.response, attempt, base_delay)
            logger.info(
                "Attempt %d/%d failed (HTTP %d), retrying in %.1fs",
                attempt + 1, retries, exc.response.status_code, delay,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_error = exc
            delay = _backoff(attempt, base_delay)
            logger.info(
                "Attempt %d/%d failed (%s), retrying in %.1fs",
                attempt + 1, retries, type(exc).__name__, delay,
            )
        else:
            break  # pragma: no cover – unreachable but keeps linter happy

        if attempt < retries - 1:
            await asyncio.sleep(delay)

    raise RetriesExhausted(retries, last_error)  # type: ignore[arg-type]


def _compute_delay(
    response: httpx.Response,
    attempt: int,
    base_delay: float,
) -> float:
    """Compute wait time, honouring ``Retry-After`` when present."""
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(float(retry_after), 0)
        except ValueError:
            pass
    return _backoff(attempt, base_delay)


def _backoff(attempt: int, base_delay: float) -> float:
    """Exponential backoff with jitter."""
    delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0, delay * 0.25)  # noqa: S311
    return delay + jitter
