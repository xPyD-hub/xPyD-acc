"""Tests for benchmark module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from xpyd_acc.benchmark import BenchmarkResult, run_benchmark


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_compute_stats_empty(self) -> None:
        result = BenchmarkResult(url="http://x", model="m", requests=0, concurrency=1)
        stats = result.compute_stats()
        assert stats.count == 0
        assert stats.mean_ms == 0.0

    def test_compute_stats_single(self) -> None:
        result = BenchmarkResult(
            url="http://x", model="m", requests=1, concurrency=1,
            latencies_ms=[100.0],
        )
        stats = result.compute_stats()
        assert stats.count == 1
        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0
        assert stats.mean_ms == 100.0
        assert stats.p50_ms == 100.0

    def test_compute_stats_multiple(self) -> None:
        lats = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        result = BenchmarkResult(
            url="http://x", model="m", requests=10, concurrency=1,
            latencies_ms=lats,
        )
        stats = result.compute_stats()
        assert stats.count == 10
        assert stats.min_ms == 10.0
        assert stats.max_ms == 100.0
        assert abs(stats.mean_ms - 55.0) < 0.01
        assert stats.p50_ms > 0

    def test_to_json(self) -> None:
        result = BenchmarkResult(
            url="http://x", model="m", requests=1, concurrency=1,
            latencies_ms=[50.0],
        )
        result.compute_stats()
        data = json.loads(result.to_json())
        assert data["url"] == "http://x"
        assert data["stats"]["count"] == 1
        assert data["latencies_ms"] == [50.0]

    def test_compute_stats_with_errors(self) -> None:
        result = BenchmarkResult(
            url="http://x", model="m", requests=3, concurrency=1,
            latencies_ms=[50.0], error_count=2,
        )
        stats = result.compute_stats()
        assert stats.count == 1
        assert stats.errors == 2


@pytest.mark.asyncio
async def test_run_benchmark_success() -> None:
    """Test run_benchmark with mocked HTTP responses."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = lambda: None

    with patch("xpyd_acc.benchmark.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await run_benchmark(
            url="http://localhost:8000",
            prompt="Hi",
            requests=3,
            concurrency=2,
        )

    assert result.requests == 3
    assert result.concurrency == 2
    assert len(result.latencies_ms) == 3
    assert result.error_count == 0
    assert result.stats is not None
    assert result.stats.count == 3


@pytest.mark.asyncio
async def test_run_benchmark_json_export(tmp_path) -> None:
    """Test JSON export."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = lambda: None

    json_file = str(tmp_path / "bench.json")

    with patch("xpyd_acc.benchmark.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        await run_benchmark(
            url="http://localhost:8000",
            requests=2,
            json_path=json_file,
        )

    with open(json_file) as f:
        data = json.load(f)
    assert data["requests"] == 2
    assert "stats" in data


@pytest.mark.asyncio
async def test_run_benchmark_with_errors() -> None:
    """Test benchmark handles request errors gracefully."""
    with patch("xpyd_acc.benchmark.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("fail"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await run_benchmark(
            url="http://localhost:8000",
            requests=2,
            concurrency=1,
        )

    assert result.error_count == 2
    assert len(result.latencies_ms) == 0
    assert result.stats is not None
    assert result.stats.count == 0
