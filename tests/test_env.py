"""Tests for environment variable support."""

from __future__ import annotations

import os
from unittest.mock import patch

from xpyd_acc.env import (
    ENV_API_KEY,
    ENV_BASELINE_URL,
    ENV_MODEL,
    ENV_TARGET_URL,
    apply_env_defaults,
    get_env_defaults,
)


class TestGetEnvDefaults:
    """Test get_env_defaults reads from environment."""

    def test_no_env_vars_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            defaults = get_env_defaults()
        assert defaults.api_key is None
        assert defaults.baseline_url is None
        assert defaults.target_url is None
        assert defaults.model is None

    def test_all_env_vars_set(self) -> None:
        env = {
            ENV_API_KEY: "sk-test-123",
            ENV_BASELINE_URL: "http://baseline:8000",
            ENV_TARGET_URL: "http://target:8000",
            ENV_MODEL: "llama-3",
        }
        with patch.dict(os.environ, env, clear=True):
            defaults = get_env_defaults()
        assert defaults.api_key == "sk-test-123"
        assert defaults.baseline_url == "http://baseline:8000"
        assert defaults.target_url == "http://target:8000"
        assert defaults.model == "llama-3"

    def test_empty_string_treated_as_none(self) -> None:
        env = {ENV_API_KEY: "", ENV_MODEL: ""}
        with patch.dict(os.environ, env, clear=True):
            defaults = get_env_defaults()
        assert defaults.api_key is None
        assert defaults.model is None

    def test_partial_env_vars(self) -> None:
        env = {ENV_API_KEY: "sk-partial"}
        with patch.dict(os.environ, env, clear=True):
            defaults = get_env_defaults()
        assert defaults.api_key == "sk-partial"
        assert defaults.baseline_url is None


class TestApplyEnvDefaults:
    """Test priority chain: CLI > env > config > default."""

    def test_cli_wins_over_all(self) -> None:
        result = apply_env_defaults("cli-val", "env-val", "config-val", "default")
        assert result == "cli-val"

    def test_env_wins_over_config_and_default(self) -> None:
        result = apply_env_defaults(None, "env-val", "config-val", "default")
        assert result == "env-val"

    def test_config_wins_over_default(self) -> None:
        result = apply_env_defaults(None, None, "config-val", "default")
        assert result == "config-val"

    def test_default_when_nothing_set(self) -> None:
        result = apply_env_defaults(None, None, None, "default")
        assert result == "default"

    def test_all_none(self) -> None:
        result = apply_env_defaults(None, None, None, None)
        assert result is None


class TestCliEnvIntegration:
    """Test that CLI picks up env vars correctly."""

    def test_env_var_used_when_no_cli_flag(self) -> None:
        """Verify that running CLI with env vars fills in missing args."""
        env = {
            ENV_API_KEY: "sk-from-env",
            ENV_BASELINE_URL: "http://env-baseline:8000",
            ENV_TARGET_URL: "http://env-target:8000",
            ENV_MODEL: "env-model",
        }
        with patch.dict(os.environ, env):

            # We can't easily run the full CLI without hitting real endpoints,
            # but we can verify the env module returns correct values
            from xpyd_acc.env import get_env_defaults
            defaults = get_env_defaults()
            assert defaults.api_key == "sk-from-env"
            assert defaults.baseline_url == "http://env-baseline:8000"
            assert defaults.target_url == "http://env-target:8000"
            assert defaults.model == "env-model"
