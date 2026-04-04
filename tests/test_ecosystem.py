"""Tests for xPyD ecosystem integration module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from xpyd_acc.ecosystem import (
    EndpointInfo,
    EndpointType,
    ProxyConfig,
    detect_endpoint_type,
    format_detect_report,
)


class TestEndpointInfo:
    def test_defaults(self) -> None:
        info = EndpointInfo(url="http://localhost:8000")
        assert info.endpoint_type == EndpointType.UNKNOWN
        assert info.model is None
        assert info.version is None
        assert info.metadata == {}

    def test_is_aggregated(self) -> None:
        info = EndpointInfo(url="http://x", endpoint_type=EndpointType.AGGREGATED)
        assert info.is_aggregated
        assert not info.is_disaggregated

    def test_is_disaggregated(self) -> None:
        info = EndpointInfo(url="http://x", endpoint_type=EndpointType.DISAGGREGATED)
        assert info.is_disaggregated
        assert not info.is_aggregated


class TestDetectEndpointType:
    @pytest.mark.asyncio
    async def test_unreachable_endpoint(self) -> None:
        """Unreachable endpoint returns UNKNOWN."""
        with patch("xpyd_acc.ecosystem.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            info = await detect_endpoint_type("http://localhost:9999")
            assert info.endpoint_type == EndpointType.UNKNOWN

    @pytest.mark.asyncio
    async def test_aggregated_from_health(self) -> None:
        """Endpoint with /health but no /info is classified as aggregated."""
        with patch("xpyd_acc.ecosystem.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            health_resp = httpx.Response(200, json={"status": "ok"})
            info_resp = httpx.Response(404)

            async def mock_get(url: str) -> httpx.Response:
                if "/health" in url:
                    return health_resp
                return info_resp

            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            info = await detect_endpoint_type("http://localhost:8000")
            assert info.endpoint_type == EndpointType.AGGREGATED

    @pytest.mark.asyncio
    async def test_disaggregated_from_info(self) -> None:
        """Endpoint with type=disaggregated in /info response."""
        with patch("xpyd_acc.ecosystem.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            health_resp = httpx.Response(200, json={"status": "ok"})
            info_data = {"type": "disaggregated", "version": "1.2.0"}
            info_resp = httpx.Response(200, json=info_data)

            async def mock_get(url: str) -> httpx.Response:
                if "/health" in url:
                    return health_resp
                if "/info" in url and "/v1/" not in url:
                    return info_resp
                return httpx.Response(404)

            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            info = await detect_endpoint_type("http://localhost:8000")
            assert info.endpoint_type == EndpointType.DISAGGREGATED
            assert info.version == "1.2.0"

    @pytest.mark.asyncio
    async def test_disaggregated_from_pd_mode(self) -> None:
        """Endpoint with mode=pd is classified as disaggregated."""
        with patch("xpyd_acc.ecosystem.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            health_resp = httpx.Response(200, json={"status": "ok"})
            info_resp = httpx.Response(200, json={"mode": "pd"})

            async def mock_get(url: str) -> httpx.Response:
                if "/health" in url:
                    return health_resp
                if "/info" in url and "/v1/" not in url:
                    return info_resp
                return httpx.Response(404)

            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            info = await detect_endpoint_type("http://localhost:8000")
            assert info.endpoint_type == EndpointType.DISAGGREGATED

    @pytest.mark.asyncio
    async def test_model_extraction_from_v1_models(self) -> None:
        """Model name extracted from /v1/models when /health fails."""
        with patch("xpyd_acc.ecosystem.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            models_resp = httpx.Response(
                200,
                json={"data": [{"id": "llama-7b"}]},
            )

            async def mock_get(url: str) -> httpx.Response:
                if "/health" in url:
                    return httpx.Response(503)
                if "/v1/models" in url:
                    return models_resp
                return httpx.Response(404)

            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            info = await detect_endpoint_type("http://localhost:8000")
            assert info.model == "llama-7b"


class TestProxyConfig:
    def test_from_dict_endpoints_format(self) -> None:
        data = {
            "endpoints": [
                {"name": "baseline", "url": "http://agg:8000", "type": "aggregated"},
                {"name": "target", "url": "http://pd:8000", "type": "disaggregated"},
            ]
        }
        config = ProxyConfig.from_dict(data)
        assert len(config.endpoints) == 2
        assert config.endpoints[0].endpoint_type == EndpointType.AGGREGATED
        assert config.endpoints[1].endpoint_type == EndpointType.DISAGGREGATED

    def test_from_dict_prefill_decode_format(self) -> None:
        data = {"prefill_url": "http://p:8000", "decode_url": "http://d:8000"}
        config = ProxyConfig.from_dict(data)
        assert len(config.endpoints) == 2
        assert config.endpoints[0].name == "prefill"
        assert config.endpoints[1].name == "decode"
        assert all(ep.endpoint_type == EndpointType.DISAGGREGATED for ep in config.endpoints)

    def test_from_dict_upstream_format(self) -> None:
        data = {"upstream": "http://server:8000"}
        config = ProxyConfig.from_dict(data)
        assert len(config.endpoints) == 1
        assert config.endpoints[0].url == "http://server:8000"

    def test_get_baseline_and_target(self) -> None:
        data = {
            "endpoints": [
                {"name": "baseline", "url": "http://agg:8000", "type": "aggregated"},
                {"name": "target", "url": "http://pd:8000", "type": "disaggregated"},
            ]
        }
        config = ProxyConfig.from_dict(data)
        baseline = config.get_baseline()
        target = config.get_target()
        assert baseline is not None
        assert baseline.url == "http://agg:8000"
        assert target is not None
        assert target.url == "http://pd:8000"

    def test_get_baseline_fallback(self) -> None:
        """Falls back to first endpoint when no explicit baseline."""
        data = {
            "endpoints": [
                {"name": "ep1", "url": "http://a:8000"},
                {"name": "ep2", "url": "http://b:8000"},
            ]
        }
        config = ProxyConfig.from_dict(data)
        assert config.get_baseline() is not None
        assert config.get_baseline().url == "http://a:8000"  # type: ignore[union-attr]

    def test_get_target_fallback(self) -> None:
        """Falls back to second endpoint when no explicit target."""
        data = {
            "endpoints": [
                {"name": "ep1", "url": "http://a:8000"},
                {"name": "ep2", "url": "http://b:8000"},
            ]
        }
        config = ProxyConfig.from_dict(data)
        assert config.get_target() is not None
        assert config.get_target().url == "http://b:8000"  # type: ignore[union-attr]

    def test_empty_config(self) -> None:
        config = ProxyConfig.from_dict({})
        assert config.endpoints == []
        assert config.get_baseline() is None
        assert config.get_target() is None

    def test_from_file_json(self, tmp_path: Path) -> None:
        data = {"endpoints": [{"name": "x", "url": "http://x:8000"}]}
        f = tmp_path / "config.json"
        f.write_text(json.dumps(data))
        config = ProxyConfig.from_file(f)
        assert len(config.endpoints) == 1


class TestFormatDetectReport:
    def test_basic_report(self) -> None:
        info = EndpointInfo(
            url="http://localhost:8000",
            endpoint_type=EndpointType.AGGREGATED,
            model="llama-7b",
        )
        report = format_detect_report(info)
        assert "localhost:8000" in report
        assert "aggregated" in report
        assert "llama-7b" in report

    def test_unknown_report(self) -> None:
        info = EndpointInfo(url="http://x")
        report = format_detect_report(info)
        assert "unknown" in report
