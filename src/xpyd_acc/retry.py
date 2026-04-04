"""Reusable async HTTP retry with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Callable, TypeVar

import httpx

T = TypeVar("T")

#: HTTP status codes that trigger a retry.
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


class RetriesExhausted(Exception):
    """All retry attempts failed."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"All {attempts} attempts failed. Last error: {last_error}")


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> Any:
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
        The return value of *func*.

    Raises:
        RetriesExhausted: If all attempts fail.
    """
    if retries < 1:
        retries = 1

    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                raise
            last_error = exc
            delay = _compute_delay(exc.response, attempt, base_delay)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_error = exc
            delay = _backoff(attempt, base_delay)
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
