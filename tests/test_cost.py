"""Tests for cost estimation module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from xpyd_acc.cost import (
    CostConfig,
    TokenUsage,
    UsageSummary,
    extract_usage,
    format_usage_summary,
)


class TestTokenUsage:
    def test_total_tokens(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert u.total_tokens == 150

    def test_defaults(self):
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0


class TestCostConfig:
    def test_estimate_zero_price(self):
        cfg = CostConfig()
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        assert cfg.estimate(usage) == 0.0

    def test_estimate_with_pricing(self):
        # $3 per 1M input, $15 per 1M output
        cfg = CostConfig(input_price_per_m=3.0, output_price_per_m=15.0)
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert cfg.estimate(usage) == 18.0

    def test_estimate_fractional(self):
        cfg = CostConfig(input_price_per_m=1.0, output_price_per_m=2.0)
        usage = TokenUsage(prompt_tokens=500_000, completion_tokens=250_000)
        cost = cfg.estimate(usage)
        assert abs(cost - 1.0) < 1e-9  # 0.5 + 0.5


class TestUsageSummary:
    def test_add(self):
        summary = UsageSummary()
        summary.add(TokenUsage(prompt_tokens=100, completion_tokens=50))
        summary.add(TokenUsage(prompt_tokens=200, completion_tokens=100))
        assert summary.total_prompt_tokens == 300
        assert summary.total_completion_tokens == 150
        assert summary.total_tokens == 450
        assert summary.num_requests == 2

    def test_apply_cost(self):
        summary = UsageSummary(
            total_prompt_tokens=1_000_000,
            total_completion_tokens=500_000,
            num_requests=10,
        )
        cfg = CostConfig(input_price_per_m=3.0, output_price_per_m=15.0)
        summary.apply_cost(cfg)
        assert summary.estimated_cost_usd is not None
        assert abs(summary.estimated_cost_usd - 10.5) < 1e-9

    def test_to_dict_without_cost(self):
        summary = UsageSummary(
            total_prompt_tokens=100,
            total_completion_tokens=50,
            num_requests=1,
        )
        d = summary.to_dict()
        assert d["total_prompt_tokens"] == 100
        assert d["total_completion_tokens"] == 50
        assert d["total_tokens"] == 150
        assert d["num_requests"] == 1
        assert "estimated_cost_usd" not in d

    def test_to_dict_with_cost(self):
        summary = UsageSummary(
            total_prompt_tokens=100,
            total_completion_tokens=50,
            num_requests=1,
            estimated_cost_usd=0.001234,
        )
        d = summary.to_dict()
        assert d["estimated_cost_usd"] == 0.001234

    def test_to_json(self):
        summary = UsageSummary(
            total_prompt_tokens=100,
            total_completion_tokens=50,
            num_requests=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        summary.to_json(path)
        data = json.loads(Path(path).read_text())
        assert data["total_tokens"] == 150
        Path(path).unlink()


class TestExtractUsage:
    def test_with_usage(self):
        resp = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 10,
            },
        }
        u = extract_usage(resp)
        assert u.prompt_tokens == 42
        assert u.completion_tokens == 10

    def test_missing_usage(self):
        resp = {"choices": [{"message": {"content": "hi"}}]}
        u = extract_usage(resp)
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0

    def test_null_usage(self):
        resp = {"usage": None}
        u = extract_usage(resp)
        assert u.total_tokens == 0

    def test_partial_usage(self):
        resp = {"usage": {"prompt_tokens": 10}}
        u = extract_usage(resp)
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 0


class TestFormatUsageSummary:
    def test_format_without_cost(self):
        summary = UsageSummary(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            num_requests=5,
        )
        text = format_usage_summary(summary)
        assert "1,000" in text
        assert "500" in text
        assert "$" not in text

    def test_format_with_cost(self):
        summary = UsageSummary(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            num_requests=5,
            estimated_cost_usd=0.0123,
        )
        text = format_usage_summary(summary)
        assert "$" in text
        assert "0.0123" in text
