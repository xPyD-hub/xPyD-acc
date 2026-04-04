"""Tests for the healthcheck module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xpyd_acc.healthcheck import (
    HealthCheckResult,
    check_endpoint,
    check_endpoints,
    format_healthcheck,
)


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_healthy_result(self) -> None:
        r = HealthCheckResult(
            url="http://localhost:8000",
            reachable=True,
            response_time_ms=42.5,
            models_available=["llama"],
            error=None,
        )
        assert r.healthy is True

    def test_unreachable_result(self) -> None:
        r = HealthCheckResult(
            url="http://localhost:8000",
            reachable=False,
            response_time_ms=None,
            models_available=[],
            error="Connection failed",
        )
        assert r.healthy is False

    def test_reachable_with_error(self) -> None:
        r = HealthCheckResult(
            url="http://localhost:8000",
            reachable=True,
            response_time_ms=100.0,
            models_available=[],
            error="HTTP 401: Unauthorized",
        )
        assert r.healthy is False


class TestCheckEndpoint:
    """Test check_endpoint function."""

    @pytest.mark.asyncio
    async def test_healthy_endpoint(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": "model-a"}, {"id": "model-b"}],
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("xpyd_acc.healthcheck.httpx.AsyncClient", return_value=mock_client):
            result = await check_endpoint("http://localhost:8000")

        assert result.reachable is True
        assert result.healthy is True
        assert result.models_available == ["model-a", "model-b"]
        assert result.response_time_ms is not None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("xpyd_acc.healthcheck.httpx.AsyncClient", return_value=mock_client):
            result = await check_endpoint("http://localhost:9999")

        assert result.reachable is False
        assert result.healthy is False
        assert "Connection failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("xpyd_acc.healthcheck.httpx.AsyncClient", return_value=mock_client):
            result = await check_endpoint("http://localhost:8000", timeout=5.0)

        assert result.reachable is False
        assert "Timeout" in (result.error or "")

    @pytest.mark.asyncio
    async def test_trailing_slash_stripped(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("xpyd_acc.healthcheck.httpx.AsyncClient", return_value=mock_client):
            result = await check_endpoint("http://localhost:8000/")

        assert result.url == "http://localhost:8000"


class TestCheckEndpoints:
    """Test check_endpoints function."""

    @pytest.mark.asyncio
    async def test_multiple_endpoints(self) -> None:
        with patch("xpyd_acc.healthcheck.check_endpoint", new_callable=AsyncMock) as mock_check:
            mock_check.side_effect = [
                HealthCheckResult("http://a", True, 10.0, ["m1"], None),
                HealthCheckResult("http://b", False, None, [], "Connection failed"),
            ]
            results = await check_endpoints(["http://a", "http://b"])

        assert len(results) == 2
        assert results[0].healthy is True
        assert results[1].healthy is False


class TestFormatHealthcheck:
    """Test format_healthcheck function."""

    def test_all_healthy(self) -> None:
        results = [
            HealthCheckResult("http://a", True, 15.0, ["model-x"], None),
            HealthCheckResult("http://b", True, 20.0, ["model-y"], None),
        ]
        text = format_healthcheck(results)
        assert "✅" in text
        assert "All endpoints healthy" in text
        assert "http://a" in text
        assert "http://b" in text

    def test_some_unhealthy(self) -> None:
        results = [
            HealthCheckResult("http://a", True, 15.0, ["m"], None),
            HealthCheckResult("http://b", False, None, [], "Connection failed"),
        ]
        text = format_healthcheck(results)
        assert "❌" in text
        assert "Some endpoints unhealthy" in text

    def test_models_listed(self) -> None:
        results = [
            HealthCheckResult("http://a", True, 10.0, ["llama", "gpt"], None),
        ]
        text = format_healthcheck(results)
        assert "llama" in text
        assert "gpt" in text

    def test_error_displayed(self) -> None:
        results = [
            HealthCheckResult("http://a", True, 50.0, [], "HTTP 401: Unauthorized"),
        ]
        text = format_healthcheck(results)
        assert "HTTP 401" in text


class TestCLIIntegration:
    """Test CLI integration for healthcheck subcommand."""

    def test_healthcheck_help(self) -> None:
        from xpyd_acc.cli import main

        with pytest.raises(SystemExit):
            main(["healthcheck", "--help"])

    def test_batch_compare_skip_healthcheck_flag(self) -> None:
        """Verify --skip-healthcheck flag is accepted."""
        from xpyd_acc.cli import main

        with pytest.raises(SystemExit):
            main(["batch-compare", "--help"])

    def test_compare_streaming_skip_healthcheck_flag(self) -> None:
        """Verify --skip-healthcheck flag is accepted."""
        from xpyd_acc.cli import main

        with pytest.raises(SystemExit):
            main(["compare-streaming", "--help"])
