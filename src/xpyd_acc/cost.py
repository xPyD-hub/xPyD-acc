"""Cost estimation from API usage data.

Tracks prompt_tokens and completion_tokens from API responses
and estimates cost based on configurable per-token pricing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CostConfig:
    """Per-token pricing configuration.

    Prices are in USD per 1M tokens (industry standard notation).
    """

    input_price_per_m: float = 0.0
    output_price_per_m: float = 0.0

    def estimate(self, usage: TokenUsage) -> float:
        """Estimate cost in USD for given token usage."""
        input_cost = (usage.prompt_tokens / 1_000_000) * self.input_price_per_m
        output_cost = (usage.completion_tokens / 1_000_000) * self.output_price_per_m
        return input_cost + output_cost


@dataclass
class UsageSummary:
    """Aggregated usage and cost for a batch run."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    num_requests: int = 0
    estimated_cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def add(self, usage: TokenUsage) -> None:
        """Add a single request's usage."""
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.num_requests += 1

    def apply_cost(self, config: CostConfig) -> None:
        """Calculate estimated cost from a pricing config."""
        total_usage = TokenUsage(
            prompt_tokens=self.total_prompt_tokens,
            completion_tokens=self.total_completion_tokens,
        )
        self.estimated_cost_usd = config.estimate(total_usage)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        d: dict = {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "num_requests": self.num_requests,
        }
        if self.estimated_cost_usd is not None:
            d["estimated_cost_usd"] = round(self.estimated_cost_usd, 6)
        return d

    def to_json(self, path: str | Path) -> None:
        """Write usage summary to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def extract_usage(response_data: dict) -> TokenUsage:
    """Extract token usage from an OpenAI-compatible API response.

    Args:
        response_data: Parsed JSON response body.

    Returns:
        TokenUsage with prompt and completion token counts.
        Returns zero counts if usage field is missing.
    """
    usage = response_data.get("usage")
    if not usage or not isinstance(usage, dict):
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )


def format_usage_summary(summary: UsageSummary) -> str:
    """Format a usage summary for terminal display."""
    lines = [
        "Token Usage Summary",
        f"  Requests:          {summary.num_requests:,}",
        f"  Prompt tokens:     {summary.total_prompt_tokens:,}",
        f"  Completion tokens: {summary.total_completion_tokens:,}",
        f"  Total tokens:      {summary.total_tokens:,}",
    ]
    if summary.estimated_cost_usd is not None:
        lines.append(f"  Estimated cost:    ${summary.estimated_cost_usd:.4f}")
    return "\n".join(lines)
