"""Tests for xpyd_acc.retry module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from xpyd_acc.retry import (
    RETRYABLE_STATUS_CODES,
    RetriesExhausted,
    RetryResult,
    RetryStats,
    retry_async,
)


def _make_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://test")
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError(f"HTTP {status}", request=req, response=resp)


def _make_error_with_retry_after(
    status: int, retry_after: str,
) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://test")
    resp = httpx.Response(status, request=req, headers={"Retry-After": retry_after})
    return httpx.HTTPStatusError(f"HTTP {status}", request=req, response=resp)


@pytest.mark.asyncio
async def test_retry_succeeds_first_try() -> None:
    func = AsyncMock(return_value="ok")
    result = await retry_async(func, retries=3, base_delay=0.01)
    assert isinstance(result, RetryResult)
    assert result.value == "ok"
    assert result.attempts == 1
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_timeout() -> None:
    func = AsyncMock(side_effect=[httpx.TimeoutException("timeout"), "ok"])
    result = await retry_async(func, retries=3, base_delay=0.01)
    assert result.value == "ok"
    assert result.attempts == 2
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_retry_succeeds_after_connect_error() -> None:
    func = AsyncMock(side_effect=[httpx.ConnectError("refused"), "ok"])
    result = await retry_async(func, retries=3, base_delay=0.01)
    assert result.value == "ok"
    assert result.attempts == 2
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_429_status() -> None:
    func = AsyncMock(side_effect=[_make_error(429), "ok"])
    result = await retry_async(func, retries=3, base_delay=0.01)
    assert result.value == "ok"
    assert result.attempts == 2
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_502_status() -> None:
    func = AsyncMock(side_effect=[_make_error(502), "ok"])
    result = await retry_async(func, retries=3, base_delay=0.01)
    assert result.value == "ok"


@pytest.mark.asyncio
async def test_retry_does_not_retry_400() -> None:
    func = AsyncMock(side_effect=_make_error(400))
    with pytest.raises(httpx.HTTPStatusError):
        await retry_async(func, retries=3, base_delay=0.01)
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_retry_exhausted() -> None:
    func = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    with pytest.raises(RetriesExhausted) as exc_info:
        await retry_async(func, retries=3, base_delay=0.01)
    assert exc_info.value.attempts == 3
    assert func.call_count == 3


@pytest.mark.asyncio
async def test_retry_respects_retry_after_header() -> None:
    err = _make_error_with_retry_after(429, "0.01")
    func = AsyncMock(side_effect=[err, "ok"])
    # large base_delay proves Retry-After overrides backoff
    result = await retry_async(func, retries=3, base_delay=100.0)
    assert result.value == "ok"
    assert result.attempts == 2
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_retryable_status_codes_coverage() -> None:
    assert 429 in RETRYABLE_STATUS_CODES
    assert 502 in RETRYABLE_STATUS_CODES
    assert 503 in RETRYABLE_STATUS_CODES
    assert 504 in RETRYABLE_STATUS_CODES
    assert 400 not in RETRYABLE_STATUS_CODES


# --- RetryResult tests ---


def test_retry_result_fields() -> None:
    r = RetryResult(value="hello", attempts=3)
    assert r.value == "hello"
    assert r.attempts == 3


# --- RetryStats tests ---


def test_retry_stats_record_no_retries() -> None:
    stats = RetryStats()
    stats.record(RetryResult(value=None, attempts=1))
    stats.record(RetryResult(value=None, attempts=1))
    assert stats.total_requests == 2
    assert stats.total_retries == 0
    assert stats.max_retries_single == 0
    assert stats.retried_request_count == 0


def test_retry_stats_record_with_retries() -> None:
    stats = RetryStats()
    stats.record(RetryResult(value=None, attempts=1))
    stats.record(RetryResult(value=None, attempts=3))  # 2 retries
    stats.record(RetryResult(value=None, attempts=2))  # 1 retry
    assert stats.total_requests == 3
    assert stats.total_retries == 3  # 0 + 2 + 1
    assert stats.max_retries_single == 2
    assert stats.retried_request_count == 2


def test_retry_stats_to_dict() -> None:
    stats = RetryStats(
        total_requests=5, total_retries=3,
        max_retries_single=2, retried_request_count=2,
    )
    d = stats.to_dict()
    assert d == {
        "total_requests": 5,
        "total_retries": 3,
        "max_retries_single": 2,
        "retried_request_count": 2,
    }


def test_retry_stats_from_dict() -> None:
    d = {
        "total_requests": 10, "total_retries": 4,
        "max_retries_single": 3, "retried_request_count": 2,
    }
    stats = RetryStats.from_dict(d)
    assert stats.total_requests == 10
    assert stats.total_retries == 4
    assert stats.max_retries_single == 3
    assert stats.retried_request_count == 2


def test_retry_stats_from_dict_defaults() -> None:
    stats = RetryStats.from_dict({})
    assert stats.total_requests == 0
    assert stats.total_retries == 0


def test_retry_stats_round_trip() -> None:
    original = RetryStats(
        total_requests=7, total_retries=2,
        max_retries_single=1, retried_request_count=2,
    )
    restored = RetryStats.from_dict(original.to_dict())
    assert restored == original
