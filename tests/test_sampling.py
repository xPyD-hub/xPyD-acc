"""Tests for M24: Sampling Parameter Support."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xpyd_acc.sampling import SamplingParams


class TestSamplingParams:
    """Tests for SamplingParams dataclass."""

    def test_defaults_are_none(self) -> None:
        sp = SamplingParams()
        assert sp.temperature is None
        assert sp.top_p is None
        assert sp.seed is None

    def test_to_payload_empty(self) -> None:
        sp = SamplingParams()
        assert sp.to_payload() == {}

    def test_to_payload_all_set(self) -> None:
        sp = SamplingParams(temperature=0.0, top_p=0.9, seed=42)
        payload = sp.to_payload()
        assert payload == {"temperature": 0.0, "top_p": 0.9, "seed": 42}

    def test_to_payload_partial(self) -> None:
        sp = SamplingParams(temperature=0.5)
        assert sp.to_payload() == {"temperature": 0.5}

    def test_from_args(self) -> None:
        args = MagicMock()
        args.temperature = 0.7
        args.top_p = 0.95
        args.seed = 123
        sp = SamplingParams.from_args(args)
        assert sp.temperature == 0.7
        assert sp.top_p == 0.95
        assert sp.seed == 123

    def test_from_args_missing_attrs(self) -> None:
        """from_args handles missing attributes gracefully."""
        args = MagicMock(spec=[])  # empty spec = no attributes
        sp = SamplingParams.from_args(args)
        assert sp.temperature is None
        assert sp.top_p is None
        assert sp.seed is None


class TestLogprobsSamplingParams:
    """Test that LogprobsCollector passes sampling params to payload."""

    @pytest.mark.asyncio
    async def test_collect_includes_sampling_params(self) -> None:
        from xpyd_acc.logprobs import LogprobsCollector

        collector = LogprobsCollector("http://localhost:8000", model="test")
        sampling = SamplingParams(temperature=0.0, seed=42)

        captured_payload: dict = {}

        async def mock_post(url, json=None, headers=None):
            captured_payload.update(json)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "model": "test",
                "choices": [
                    {
                        "message": {"content": "hello"},
                        "logprobs": {"content": []},
                    }
                ],
            }
            return resp

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await collector.collect("test", sampling_params=sampling, retries=0)

        assert captured_payload["temperature"] == 0.0
        assert captured_payload["seed"] == 42
        assert "top_p" not in captured_payload

    @pytest.mark.asyncio
    async def test_collect_no_sampling_params(self) -> None:
        from xpyd_acc.logprobs import LogprobsCollector

        collector = LogprobsCollector("http://localhost:8000", model="test")
        captured_payload: dict = {}

        async def mock_post(url, json=None, headers=None):
            captured_payload.update(json)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "model": "test",
                "choices": [
                    {
                        "message": {"content": "hello"},
                        "logprobs": {"content": []},
                    }
                ],
            }
            return resp

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await collector.collect("test", retries=0)

        assert "temperature" not in captured_payload
        assert "seed" not in captured_payload


class TestConfigSamplingParams:
    """Test TOML config support for sampling params."""

    def test_defaults_config_has_sampling_fields(self) -> None:
        from xpyd_acc.config import DefaultsConfig

        cfg = DefaultsConfig()
        assert cfg.temperature is None
        assert cfg.top_p is None
        assert cfg.seed is None

    def test_load_config_with_sampling(self, tmp_path) -> None:
        from xpyd_acc.config import load_config

        toml_content = """
[defaults]
temperature = 0.0
top_p = 0.95
seed = 42
"""
        cfg_file = tmp_path / "xpyd-acc.toml"
        cfg_file.write_text(toml_content)
        config = load_config(cfg_file)
        assert config.defaults.temperature == 0.0
        assert config.defaults.top_p == 0.95
        assert config.defaults.seed == 42

    def test_merge_cli_args_sampling(self) -> None:
        from xpyd_acc.config import AppConfig, DefaultsConfig, merge_cli_args

        config = AppConfig(
            defaults=DefaultsConfig(temperature=0.5, top_p=0.9, seed=100),
        )
        args = {
            "temperature": None,
            "top_p": 0.8,  # CLI override
            "seed": None,
            "command": "compare-logprobs",
        }
        merged = merge_cli_args(config, args, "compare-logprobs")
        assert merged["temperature"] == 0.5  # from config
        assert merged["top_p"] == 0.8  # CLI wins
        assert merged["seed"] == 100  # from config


class TestEnvSamplingParams:
    """Test environment variable support for sampling params."""

    def test_env_defaults_sampling(self) -> None:
        from xpyd_acc.env import get_env_defaults

        with patch.dict(os.environ, {
            "XPYD_ACC_TEMPERATURE": "0.3",
            "XPYD_ACC_TOP_P": "0.85",
            "XPYD_ACC_SEED": "99",
        }):
            env = get_env_defaults()
            assert env.temperature == 0.3
            assert env.top_p == 0.85
            assert env.seed == 99

    def test_env_defaults_no_sampling(self) -> None:
        from xpyd_acc.env import get_env_defaults

        with patch.dict(os.environ, {}, clear=True):
            env = get_env_defaults()
            assert env.temperature is None
            assert env.top_p is None
            assert env.seed is None


class TestCLISamplingArgs:
    """Test CLI argument parsing for sampling params."""

    def test_compare_logprobs_sampling_args(self) -> None:
        # Just verify parsing doesn't fail
        from unittest.mock import patch as _patch

        from xpyd_acc.cli import main

        with _patch("xpyd_acc.cli.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock()
            main([
                "compare-logprobs",
                "--baseline", "http://a",
                "--target", "http://b",
                "--prompt", "test",
                "--temperature", "0",
                "--top-p", "0.9",
                "--seed", "42",
            ])
            # If we get here, parsing succeeded
            assert mock_asyncio.run.called
