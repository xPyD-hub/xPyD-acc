"""Webhook notification support for divergence alerts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from xpyd_acc.log import get_logger
from xpyd_acc.retry import retry_async

logger = get_logger("notify")


@dataclass
class WebhookConfig:
    """Configuration for webhook notifications."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    always: bool = False


@dataclass
class WebhookPayload:
    """Payload sent to webhook endpoint."""

    event: str  # "batch_complete" or "watch_divergence"
    divergence_detected: bool
    total_samples: int
    divergent_samples: int
    divergence_rate: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event": self.event,
            "divergence_detected": self.divergence_detected,
            "total_samples": self.total_samples,
            "divergent_samples": self.divergent_samples,
            "divergence_rate": self.divergence_rate,
            **self.extra,
        }


async def send_webhook(
    config: WebhookConfig,
    payload: WebhookPayload,
    *,
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 30.0,
) -> bool:
    """Send webhook notification. Returns True on success."""
    if not config.always and not payload.divergence_detected:
        logger.debug("No divergence and --webhook-always not set; skipping webhook")
        return False

    data = payload.to_dict()
    logger.info("Sending webhook to %s (event=%s)", config.url, payload.event)

    async def _do_post() -> bool:
        headers = {"Content-Type": "application/json", **config.headers}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(config.url, content=json.dumps(data), headers=headers)
            resp.raise_for_status()
            logger.info("Webhook sent successfully (status=%d)", resp.status_code)
            return True

    try:
        result = await retry_async(_do_post, retries=retries, base_delay=retry_delay)
        return result.value
    except Exception:
        logger.error("Webhook delivery failed after %d retries", retries)
        return False


def parse_webhook_headers(header_strings: list[str] | None) -> dict[str, str]:
    """Parse 'Key: Value' strings into a dict."""
    if not header_strings:
        return {}
    result: dict[str, str] = {}
    for h in header_strings:
        if ":" not in h:
            logger.warning("Ignoring malformed header (no colon): %s", h)
            continue
        key, value = h.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def load_webhook_config_from_toml(config: dict[str, Any]) -> WebhookConfig | None:
    """Extract webhook config from parsed TOML config dict."""
    notifications = config.get("notifications", {})
    url = notifications.get("webhook_url")
    if not url:
        return None
    headers = notifications.get("webhook_headers", {})
    always = notifications.get("webhook_always", False)
    return WebhookConfig(url=url, headers=headers, always=always)


def resolve_webhook_config(
    *,
    cli_url: str | None = None,
    cli_headers: list[str] | None = None,
    cli_always: bool = False,
    env_url: str | None = None,
    toml_config: dict[str, Any] | None = None,
) -> WebhookConfig | None:
    """Resolve webhook config from CLI > env > TOML priority chain."""
    url = cli_url or env_url
    toml_wh = load_webhook_config_from_toml(toml_config) if toml_config else None

    if not url and toml_wh:
        url = toml_wh.url

    if not url:
        return None

    headers: dict[str, str] = {}
    always = cli_always

    if toml_wh:
        headers.update(toml_wh.headers)
        if not cli_always:
            always = toml_wh.always

    # CLI headers override TOML headers
    if cli_headers:
        headers.update(parse_webhook_headers(cli_headers))

    return WebhookConfig(url=url, headers=headers, always=always)
