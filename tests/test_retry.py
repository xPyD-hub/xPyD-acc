"""Tests for xpyd_acc.retry module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from xpyd_acc.retry import RETRYABLE_STATUS_CODES, RetriesExhausted, retry_async


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
    assert result == "ok"
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_timeout() -> None:
    func = AsyncMock(side_effect=[httpx.TimeoutException("timeout"), "ok"])
    result = await retry_async(func, retries=3, base_delay=0.01)
    assert result == "ok"
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_retry_succeeds_after_connect_error() -> None:
    func = AsyncMock(side_effect=[httpx.ConnectError("refused"), "ok"])
    result = await retry_async(func, retries=3, base_delay=0.01)
    assert result == "ok"
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_429_status() -> None:
    func = AsyncMock(side_effect=[_make_error(429), "ok"])
    result = await retry_async(func, retries=3, base_delay=0.01)
    assert result == "ok"
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_502_status() -> None:
    func = AsyncMock(side_effect=[_make_error(502), "ok"])
    result = await retry_async(func, retries=3, base_delay=0.01)
    assert result == "ok"


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
    assert result == "ok"
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_retryable_status_codes_coverage() -> None:
    assert 429 in RETRYABLE_STATUS_CODES
    assert 502 in RETRYABLE_STATUS_CODES
    assert 503 in RETRYABLE_STATUS_CODES
    assert 504 in RETRYABLE_STATUS_CODES
    assert 400 not in RETRYABLE_STATUS_CODES
