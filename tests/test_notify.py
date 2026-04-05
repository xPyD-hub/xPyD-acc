"""Tests for webhook notification support (M41)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from xpyd_acc.notify import (
    WebhookConfig,
    WebhookPayload,
    load_webhook_config_from_toml,
    parse_webhook_headers,
    resolve_webhook_config,
    send_webhook,
)
from xpyd_acc.retry import RetryResult


def _make_payload(*, divergence: bool = True) -> WebhookPayload:
    return WebhookPayload(
        event="batch_complete",
        divergence_detected=divergence,
        total_samples=100,
        divergent_samples=5 if divergence else 0,
        divergence_rate=0.05 if divergence else 0.0,
    )


class TestWebhookPayload:
    def test_to_dict(self) -> None:
        p = _make_payload()
        d = p.to_dict()
        assert d["event"] == "batch_complete"
        assert d["divergence_detected"] is True
        assert d["total_samples"] == 100
        assert d["divergent_samples"] == 5
        assert d["divergence_rate"] == 0.05

    def test_to_dict_with_extra(self) -> None:
        p = _make_payload()
        p.extra = {"run_id": "abc123"}
        d = p.to_dict()
        assert d["run_id"] == "abc123"


class TestParseWebhookHeaders:
    def test_empty(self) -> None:
        assert parse_webhook_headers(None) == {}
        assert parse_webhook_headers([]) == {}

    def test_valid_headers(self) -> None:
        result = parse_webhook_headers(["Authorization: Bearer tok", "X-Custom: val"])
        assert result == {"Authorization": "Bearer tok", "X-Custom": "val"}

    def test_malformed_header_skipped(self) -> None:
        result = parse_webhook_headers(["NoColon", "Good: Value"])
        assert result == {"Good": "Value"}


class TestLoadWebhookConfigFromToml:
    def test_no_notifications_section(self) -> None:
        assert load_webhook_config_from_toml({}) is None

    def test_no_url(self) -> None:
        assert load_webhook_config_from_toml({"notifications": {}}) is None

    def test_full_config(self) -> None:
        cfg = load_webhook_config_from_toml({
            "notifications": {
                "webhook_url": "https://example.com/hook",
                "webhook_headers": {"X-Key": "secret"},
                "webhook_always": True,
            }
        })
        assert cfg is not None
        assert cfg.url == "https://example.com/hook"
        assert cfg.headers == {"X-Key": "secret"}
        assert cfg.always is True


class TestResolveWebhookConfig:
    def test_no_config(self) -> None:
        assert resolve_webhook_config() is None

    def test_cli_url_wins(self) -> None:
        cfg = resolve_webhook_config(
            cli_url="https://cli.example.com",
            env_url="https://env.example.com",
        )
        assert cfg is not None
        assert cfg.url == "https://cli.example.com"

    def test_env_url_fallback(self) -> None:
        cfg = resolve_webhook_config(env_url="https://env.example.com")
        assert cfg is not None
        assert cfg.url == "https://env.example.com"

    def test_toml_fallback(self) -> None:
        cfg = resolve_webhook_config(
            toml_config={"notifications": {"webhook_url": "https://toml.example.com"}}
        )
        assert cfg is not None
        assert cfg.url == "https://toml.example.com"

    def test_cli_headers_override_toml(self) -> None:
        cfg = resolve_webhook_config(
            cli_url="https://example.com",
            cli_headers=["X-Key: cli-val"],
            toml_config={"notifications": {
                "webhook_url": "https://toml.example.com",
                "webhook_headers": {"X-Key": "toml-val", "X-Other": "kept"},
            }},
        )
        assert cfg is not None
        assert cfg.headers["X-Key"] == "cli-val"
        assert cfg.headers["X-Other"] == "kept"

    def test_cli_always_flag(self) -> None:
        cfg = resolve_webhook_config(cli_url="https://example.com", cli_always=True)
        assert cfg is not None
        assert cfg.always is True


class TestSendWebhook:
    @pytest.mark.asyncio
    async def test_skip_when_no_divergence_and_not_always(self) -> None:
        config = WebhookConfig(url="https://example.com/hook")
        payload = _make_payload(divergence=False)
        result = await send_webhook(config, payload)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_when_always_and_no_divergence(self) -> None:
        config = WebhookConfig(url="https://example.com/hook", always=True)
        payload = _make_payload(divergence=False)

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None

        with patch("xpyd_acc.notify.retry_async", new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = RetryResult(value=True, attempts=1)
            result = await send_webhook(config, payload)
            assert result is True
            mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_on_divergence(self) -> None:
        config = WebhookConfig(url="https://example.com/hook")
        payload = _make_payload(divergence=True)

        with patch("xpyd_acc.notify.retry_async", new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = RetryResult(value=True, attempts=1)
            result = await send_webhook(config, payload)
            assert result is True

    @pytest.mark.asyncio
    async def test_send_with_custom_headers(self) -> None:
        config = WebhookConfig(
            url="https://example.com/hook",
            headers={"Authorization": "Bearer tok123"},
        )
        payload = _make_payload()

        with patch("xpyd_acc.notify.retry_async", new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = RetryResult(value=True, attempts=1)
            result = await send_webhook(config, payload)
            assert result is True

    @pytest.mark.asyncio
    async def test_retry_failure(self) -> None:
        config = WebhookConfig(url="https://example.com/hook")
        payload = _make_payload()

        with patch("xpyd_acc.notify.retry_async", new_callable=AsyncMock) as mock_retry:
            mock_retry.side_effect = Exception("connection failed")
            result = await send_webhook(config, payload)
            assert result is False
