"""Sample filtering for batch comparison reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xpyd_acc.log import get_logger

logger = get_logger("filter")


@dataclass
class FilterConfig:
    """Configuration for filtering batch report samples."""

    classification: str | None = None
    divergent_only: bool = False
    matched_only: bool = False
    min_logprob_gap: float | None = None
    max_logprob_gap: float | None = None
    min_context_length: int | None = None
    max_context_length: int | None = None
    search: str | None = None


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a batch report JSON file."""
    with open(path) as f:
        return json.load(f)


def filter_samples(
    report: dict[str, Any],
    config: FilterConfig,
) -> dict[str, Any]:
    """Filter samples in a batch report and recalculate statistics."""
    samples = report.get("samples", [])
    filtered = _apply_filters(samples, config)

    total = len(filtered)
    divergent = sum(1 for s in filtered if not s.get("exact_match", True))
    matched = total - divergent
    rate = divergent / total if total > 0 else 0.0

    result = dict(report)
    result["samples"] = filtered
    result["total_samples"] = total
    result["divergent_samples"] = divergent
    result["match_samples"] = matched
    result["divergence_rate"] = rate

    logger.info(
        "Filtered %d -> %d samples (divergent=%d, rate=%.2f%%)",
        len(samples), total, divergent, rate * 100,
    )
    return result


def _apply_filters(
    samples: list[dict[str, Any]],
    config: FilterConfig,
) -> list[dict[str, Any]]:
    """Apply all filter criteria (AND logic)."""
    result = samples

    if config.divergent_only:
        result = [s for s in result if not s.get("exact_match", True)]

    if config.matched_only:
        result = [s for s in result if s.get("exact_match", False)]

    if config.classification is not None:
        result = [
            s for s in result
            if s.get("classification") == config.classification
        ]

    if config.min_logprob_gap is not None:
        result = [
            s for s in result
            if s.get("logprob_gap") is not None
            and s["logprob_gap"] >= config.min_logprob_gap
        ]

    if config.max_logprob_gap is not None:
        result = [
            s for s in result
            if s.get("logprob_gap") is not None
            and s["logprob_gap"] <= config.max_logprob_gap
        ]

    if config.min_context_length is not None:
        result = [
            s for s in result
            if s.get("context_length", 0) >= config.min_context_length
        ]

    if config.max_context_length is not None:
        result = [
            s for s in result
            if s.get("context_length", 0) <= config.max_context_length
        ]

    if config.search is not None:
        needle = config.search.lower()
        result = [
            s for s in result
            if needle in s.get("prompt", "").lower()
            or needle in s.get("baseline_output", "").lower()
            or needle in s.get("target_output", "").lower()
        ]

    return result


def save_report(report: dict[str, Any], path: str | Path) -> None:
    """Save a filtered report to JSON."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Filtered report saved to %s", path)
