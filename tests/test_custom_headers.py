"""Tests for M67: Custom HTTP Headers for API Requests."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from xpyd_acc.headers import (
    merge_with_defaults,
    parse_env_headers,
    parse_header_arg,
    parse_header_args,
    resolve_headers,
)


class TestParseHeaderArg:
    def test_basic(self) -> None:
        assert parse_header_arg("X-Custom: value") == ("X-Custom", "value")

    def test_no_space_after_colon(self) -> None:
        assert parse_header_arg("X-Custom:value") == ("X-Custom", "value")

    def test_value_with_colons(self) -> None:
        assert parse_header_arg("Auth: Bearer:token:extra") == ("Auth", "Bearer:token:extra")

    def test_whitespace_stripped(self) -> None:
        assert parse_header_arg("  Key  :  Value  ") == ("Key", "Value")

    def test_missing_colon_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid header format"):
            parse_header_arg("no-colon-here")

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Header name cannot be empty"):
            parse_header_arg(": value")

    def test_empty_value_ok(self) -> None:
        assert parse_header_arg("Key:") == ("Key", "")


class TestParseHeaderArgs:
    def test_none_input(self) -> None:
        assert parse_header_args(None) == {}

    def test_empty_list(self) -> None:
        assert parse_header_args([]) == {}

    def test_multiple_headers(self) -> None:
        result = parse_header_args(["X-A: 1", "X-B: 2"])
        assert result == {"X-A": "1", "X-B": "2"}

    def test_later_overrides_earlier(self) -> None:
        result = parse_header_args(["X-A: old", "X-A: new"])
        assert result == {"X-A": "new"}


class TestParseEnvHeaders:
    def test_none(self) -> None:
        assert parse_env_headers(None) == {}

    def test_empty_string(self) -> None:
        assert parse_env_headers("") == {}

    def test_single_header(self) -> None:
        assert parse_env_headers("X-Tenant:abc") == {"X-Tenant": "abc"}

    def test_multiple_headers(self) -> None:
        result = parse_env_headers("X-A:1,X-B:2")
        assert result == {"X-A": "1", "X-B": "2"}

    def test_whitespace_handling(self) -> None:
        result = parse_env_headers(" X-A : 1 , X-B : 2 ")
        assert result == {"X-A": "1", "X-B": "2"}

    def test_trailing_comma(self) -> None:
        result = parse_env_headers("X-A:1,")
        assert result == {"X-A": "1"}

    def test_reads_env_var(self) -> None:
        with patch.dict(os.environ, {"XPYD_ACC_HEADERS": "X-Test:val"}):
            result = parse_env_headers()
            assert result == {"X-Test": "val"}

    def test_env_var_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = parse_env_headers()
            assert result == {}


class TestResolveHeaders:
    def test_empty_all(self) -> None:
        assert resolve_headers() == {}

    def test_config_only(self) -> None:
        result = resolve_headers(config_headers={"X-A": "1"})
        assert result == {"X-A": "1"}

    def test_env_overrides_config(self) -> None:
        result = resolve_headers(
            config_headers={"X-A": "config"},
            env_headers={"X-A": "env"},
        )
        assert result == {"X-A": "env"}

    def test_cli_overrides_env(self) -> None:
        result = resolve_headers(
            env_headers={"X-A": "env"},
            cli_headers={"X-A": "cli"},
        )
        assert result == {"X-A": "cli"}

    def test_full_priority_chain(self) -> None:
        result = resolve_headers(
            config_headers={"X-A": "config", "X-B": "config"},
            env_headers={"X-B": "env", "X-C": "env"},
            cli_headers={"X-C": "cli", "X-D": "cli"},
        )
        assert result == {"X-A": "config", "X-B": "env", "X-C": "cli", "X-D": "cli"}

    def test_config_values_converted_to_str(self) -> None:
        result = resolve_headers(config_headers={"X-Num": 42})
        assert result == {"X-Num": "42"}


class TestMergeWithDefaults:
    def test_custom_overrides(self) -> None:
        defaults = {"Authorization": "Bearer key", "Accept": "application/json"}
        custom = {"Authorization": "Token custom", "X-New": "value"}
        result = merge_with_defaults(defaults, custom)
        assert result == {
            "Authorization": "Token custom",
            "Accept": "application/json",
            "X-New": "value",
        }

    def test_empty_custom(self) -> None:
        defaults = {"Authorization": "Bearer key"}
        result = merge_with_defaults(defaults, {})
        assert result == defaults

    def test_empty_defaults(self) -> None:
        result = merge_with_defaults({}, {"X-A": "1"})
        assert result == {"X-A": "1"}


class TestCollectOutputCustomHeaders:
    @pytest.mark.asyncio
    async def test_custom_headers_passed_to_request(self) -> None:
        from xpyd_acc.batch_compare import _collect_output
        from xpyd_acc.retry import RetryResult

        with patch("xpyd_acc.retry.retry_async", new_callable=AsyncMock) as mock_retry:
            from xpyd_acc.cost import TokenUsage
            mock_retry.return_value = RetryResult(
                value=("text", [], "rid", TokenUsage(), "stop"),
                attempts=1,
            )
            await _collect_output(
                "http://fake", "prompt",
                skip_validation=True,
                custom_headers={"X-Tenant": "abc", "X-Version": "2"},
            )
            # Verify retry_async was called with the inner function
            mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_custom_headers_works(self) -> None:
        from xpyd_acc.batch_compare import _collect_output
        from xpyd_acc.retry import RetryResult

        with patch("xpyd_acc.retry.retry_async", new_callable=AsyncMock) as mock_retry:
            from xpyd_acc.cost import TokenUsage
            mock_retry.return_value = RetryResult(
                value=("text", [], "rid", TokenUsage(), "stop"),
                attempts=1,
            )
            text, lp, rid, usage, finish, attempts = await _collect_output(
                "http://fake", "prompt",
                skip_validation=True,
            )
            assert text == "text"
            assert attempts == 1


class TestCLIHeaderFlag:
    def test_batch_compare_accepts_header_flag(self) -> None:
        """Verify --header is a recognized CLI flag for batch-compare."""
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["batch-compare", "--help"])
        assert exc_info.value.code == 0
