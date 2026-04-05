"""Custom HTTP header parsing and resolution for API requests.

Supports:
- CLI ``--header "Key: Value"`` (repeatable)
- Environment variable ``XPYD_ACC_HEADERS`` (comma-separated ``Key:Value`` pairs)
- TOML config ``[defaults] headers = {"X-Custom" = "value"}``

Priority chain: CLI > env > config (consistent with other settings).
"""

from __future__ import annotations

import os
from typing import Any


def parse_header_arg(raw: str) -> tuple[str, str]:
    """Parse a single ``Key: Value`` string into a (key, value) tuple.

    Accepts both ``Key: Value`` and ``Key:Value`` (space after colon is optional).

    Raises:
        ValueError: If *raw* does not contain a colon separator.
    """
    if ":" not in raw:
        raise ValueError(f"Invalid header format (expected 'Key: Value'): {raw!r}")
    key, value = raw.split(":", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"Header name cannot be empty: {raw!r}")
    return key, value


def parse_header_args(raw_list: list[str] | None) -> dict[str, str]:
    """Parse a list of ``--header`` CLI arguments into a dict.

    Later entries override earlier ones for the same key.
    """
    if not raw_list:
        return {}
    result: dict[str, str] = {}
    for raw in raw_list:
        key, value = parse_header_arg(raw)
        result[key] = value
    return result


def parse_env_headers(env_value: str | None = None) -> dict[str, str]:
    """Parse ``XPYD_ACC_HEADERS`` environment variable.

    Format: comma-separated ``Key:Value`` pairs.
    Example: ``X-Tenant:abc,X-Version:2``
    """
    if env_value is None:
        env_value = os.environ.get("XPYD_ACC_HEADERS")
    if not env_value:
        return {}
    result: dict[str, str] = {}
    for part in env_value.split(","):
        part = part.strip()
        if not part:
            continue
        key, value = parse_header_arg(part)
        result[key] = value
    return result


def resolve_headers(
    *,
    cli_headers: dict[str, str] | None = None,
    env_headers: dict[str, str] | None = None,
    config_headers: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Merge headers from config, env, and CLI (CLI wins).

    Priority: CLI > env > config.
    """
    merged: dict[str, str] = {}
    if config_headers:
        for k, v in config_headers.items():
            merged[str(k)] = str(v)
    if env_headers:
        merged.update(env_headers)
    if cli_headers:
        merged.update(cli_headers)
    return merged


def merge_with_defaults(
    defaults: dict[str, str],
    custom: dict[str, str],
) -> dict[str, str]:
    """Merge custom headers into defaults. Custom headers take precedence."""
    result = dict(defaults)
    result.update(custom)
    return result
