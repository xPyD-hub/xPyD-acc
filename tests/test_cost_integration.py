"""Tests for M60: Cost Tracking Integration into Batch Compare."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd_acc.batch_compare import (
    compute_report,
    format_report,
    load_report,
)
from xpyd_acc.cost import CostConfig, TokenUsage, UsageSummary


class TestBatchReportUsageField:
    def test_default_none(self):
        report = compute_report([])
        assert report.usage is None

    def test_usage_attached(self):
        report = compute_report([])
        usage = UsageSummary(
            total_prompt_tokens=100,
            total_completion_tokens=50,
            num_requests=2,
        )
        report.usage = usage
        assert report.usage is not None
        assert report.usage.total_tokens == 150


class TestBatchReportToJsonWithUsage:
    def test_usage_in_json(self):
        report = compute_report([])
        report.usage = UsageSummary(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            num_requests=5,
        )
        data = json.loads(report.to_json())
        assert data["usage"] is not None
        assert data["usage"]["total_prompt_tokens"] == 1000
        assert data["usage"]["total_completion_tokens"] == 500
        assert data["usage"]["total_tokens"] == 1500
        assert data["usage"]["num_requests"] == 5

    def test_no_usage_in_json(self):
        report = compute_report([])
        data = json.loads(report.to_json())
        assert data["usage"] is None

    def test_usage_with_cost_in_json(self):
        report = compute_report([])
        usage = UsageSummary(
            total_prompt_tokens=1_000_000,
            total_completion_tokens=500_000,
            num_requests=10,
        )
        cost_cfg = CostConfig(input_price_per_m=3.0, output_price_per_m=15.0)
        usage.apply_cost(cost_cfg)
        report.usage = usage
        data = json.loads(report.to_json())
        assert data["usage"]["estimated_cost_usd"] == 10.5


class TestLoadReportWithUsage:
    def test_round_trip(self):
        report = compute_report([])
        report.usage = UsageSummary(
            total_prompt_tokens=200,
            total_completion_tokens=100,
            num_requests=3,
            estimated_cost_usd=0.005,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        Path(path).write_text(report.to_json())
        loaded = load_report(path)
        assert loaded.usage is not None
        assert loaded.usage.total_prompt_tokens == 200
        assert loaded.usage.total_completion_tokens == 100
        assert loaded.usage.num_requests == 3
        assert loaded.usage.estimated_cost_usd == 0.005
        Path(path).unlink()

    def test_load_report_without_usage(self):
        report = compute_report([])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        Path(path).write_text(report.to_json())
        loaded = load_report(path)
        assert loaded.usage is None
        Path(path).unlink()


class TestFormatReportWithUsage:
    def test_usage_displayed(self):
        report = compute_report([])
        report.usage = UsageSummary(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            num_requests=5,
        )
        text = format_report(report)
        assert "Token Usage Summary" in text
        assert "1,000" in text
        assert "500" in text

    def test_usage_with_cost_displayed(self):
        report = compute_report([])
        usage = UsageSummary(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            num_requests=5,
        )
        usage.estimated_cost_usd = 0.0123
        report.usage = usage
        text = format_report(report)
        assert "$" in text
        assert "0.0123" in text

    def test_no_usage_no_section(self):
        report = compute_report([])
        text = format_report(report)
        assert "Token Usage" not in text


class TestMarkdownWithUsage:
    def test_usage_in_markdown(self):
        report = compute_report([])
        report.usage = UsageSummary(
            total_prompt_tokens=2000,
            total_completion_tokens=800,
            num_requests=10,
        )
        md = report.to_markdown()
        assert "## Token Usage" in md
        assert "2,000" in md
        assert "800" in md

    def test_no_usage_no_section_in_markdown(self):
        report = compute_report([])
        md = report.to_markdown()
        assert "## Token Usage" not in md


class TestCostConfigIntegration:
    def test_toml_cost_section(self):
        from xpyd_acc.config import load_config

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            f.write("[cost]\ninput_price_per_m = 3.0\noutput_price_per_m = 15.0\n")
            path = f.name
        config = load_config(path)
        assert config.cost.input_price_per_m == 3.0
        assert config.cost.output_price_per_m == 15.0
        Path(path).unlink()

    def test_toml_no_cost_section(self):
        from xpyd_acc.config import load_config

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            f.write("[defaults]\nmodel = 'test'\n")
            path = f.name
        config = load_config(path)
        assert config.cost.input_price_per_m == 0.0
        assert config.cost.output_price_per_m == 0.0
        Path(path).unlink()


class TestConfigValidateWithCost:
    def test_cost_section_valid(self):
        from xpyd_acc.config_validate import validate_config

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            f.write("[cost]\ninput_price_per_m = 3.0\noutput_price_per_m = 15.0\n")
            path = f.name
        messages = validate_config(Path(path))
        assert messages == []
        Path(path).unlink()

    def test_cost_section_unknown_key(self):
        from xpyd_acc.config_validate import validate_config

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            f.write("[cost]\nbogus_key = 1.0\n")
            path = f.name
        messages = validate_config(Path(path))
        assert any("bogus_key" in m for m in messages)
        Path(path).unlink()


class TestCollectOutputReturnsUsage:
    """Test that _collect_output returns TokenUsage as 4th element."""

    @pytest.mark.asyncio
    async def test_collect_output_returns_usage(self):
        from xpyd_acc.batch_compare import _collect_output

        with patch("xpyd_acc.retry.retry_async") as mock_retry:
            mock_retry.return_value = (
                "hello",
                [],
                "rid-123",
                TokenUsage(prompt_tokens=10, completion_tokens=5),
            )
            text, lp, rid, usage = await _collect_output(
                "http://fake", "test prompt",
                skip_validation=True,
            )
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 5
