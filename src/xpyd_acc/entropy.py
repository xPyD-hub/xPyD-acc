"""Output entropy analysis for divergence characterization."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EntropyStats:
    """Summary statistics for a sequence of entropy values."""

    min: float
    max: float
    mean: float
    median: float
    p95: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "p95": self.p95,
            "count": self.count,
        }


@dataclass
class EntropyComparison:
    """Entropy comparison at and around a divergence point."""

    divergence_index: int
    baseline_entropy: float
    target_entropy: float
    delta: float
    context_baseline: list[float] = field(default_factory=list)
    context_target: list[float] = field(default_factory=list)
    context_start: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "divergence_index": self.divergence_index,
            "baseline_entropy": self.baseline_entropy,
            "target_entropy": self.target_entropy,
            "delta": self.delta,
            "context_baseline": self.context_baseline,
            "context_target": self.context_target,
            "context_start": self.context_start,
        }


def token_entropy(logprobs: dict[str, float]) -> float:
    """Compute Shannon entropy (in nats) from a top-K logprob dict.

    Parameters
    ----------
    logprobs : dict[str, float]
        Mapping of token string to log-probability (natural log).

    Returns
    -------
    float
        Shannon entropy. Returns 0.0 for empty or single-token dicts.
    """
    if len(logprobs) <= 1:
        return 0.0

    probs = [math.exp(lp) for lp in logprobs.values()]
    total = sum(probs)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for p in probs:
        p_norm = p / total
        if p_norm > 0:
            entropy -= p_norm * math.log(p_norm)
    return entropy


def sequence_entropy(token_logprobs: list[dict[str, float]]) -> list[float]:
    """Compute per-position entropy for a sequence of logprob dicts."""
    return [token_entropy(lp) for lp in token_logprobs]


def entropy_stats(entropies: list[float]) -> EntropyStats:
    """Compute summary statistics for a list of entropy values.

    Parameters
    ----------
    entropies : list[float]
        Per-position entropy values.

    Returns
    -------
    EntropyStats
        Summary statistics. For empty list, all values are 0.0.
    """
    if not entropies:
        return EntropyStats(min=0.0, max=0.0, mean=0.0, median=0.0, p95=0.0, count=0)

    sorted_e = sorted(entropies)
    n = len(sorted_e)
    p95_idx = min(int(n * 0.95), n - 1)

    return EntropyStats(
        min=sorted_e[0],
        max=sorted_e[-1],
        mean=statistics.mean(sorted_e),
        median=statistics.median(sorted_e),
        p95=sorted_e[p95_idx],
        count=n,
    )


def entropy_at_divergence(
    baseline_logprobs: list[dict[str, float]],
    target_logprobs: list[dict[str, float]],
    divergence_index: int,
    context_window: int = 5,
) -> EntropyComparison:
    """Compare entropy at and around a divergence point.

    Parameters
    ----------
    baseline_logprobs : list[dict[str, float]]
        Per-position top-K logprobs from baseline.
    target_logprobs : list[dict[str, float]]
        Per-position top-K logprobs from target.
    divergence_index : int
        Token index where divergence was detected.
    context_window : int
        Number of positions before and after to include.

    Returns
    -------
    EntropyComparison
        Entropy data at the divergence point with context.
    """
    if divergence_index < len(baseline_logprobs):
        bl_entropy = token_entropy(baseline_logprobs[divergence_index])
    else:
        bl_entropy = 0.0
    if divergence_index < len(target_logprobs):
        tg_entropy = token_entropy(target_logprobs[divergence_index])
    else:
        tg_entropy = 0.0

    start = max(0, divergence_index - context_window)
    end_bl = min(len(baseline_logprobs), divergence_index + context_window + 1)
    end_tg = min(len(target_logprobs), divergence_index + context_window + 1)

    ctx_bl = [token_entropy(baseline_logprobs[i]) for i in range(start, end_bl)]
    ctx_tg = [token_entropy(target_logprobs[i]) for i in range(start, end_tg)]

    return EntropyComparison(
        divergence_index=divergence_index,
        baseline_entropy=bl_entropy,
        target_entropy=tg_entropy,
        delta=tg_entropy - bl_entropy,
        context_baseline=ctx_bl,
        context_target=ctx_tg,
        context_start=start,
    )


def format_entropy_stats(stats: EntropyStats) -> str:
    """Format entropy stats for terminal display."""
    lines = [
        "Entropy Statistics",
        "=" * 40,
        f"  Count:  {stats.count}",
        f"  Min:    {stats.min:.4f}",
        f"  Max:    {stats.max:.4f}",
        f"  Mean:   {stats.mean:.4f}",
        f"  Median: {stats.median:.4f}",
        f"  P95:    {stats.p95:.4f}",
    ]
    return "\n".join(lines)


def format_entropy_comparison(comp: EntropyComparison) -> str:
    """Format entropy comparison for terminal display."""
    ctx_end = comp.context_start + len(comp.context_baseline) - 1
    lines = [
        f"Entropy at divergence (index {comp.divergence_index})",
        "=" * 50,
        f"  Baseline entropy: {comp.baseline_entropy:.4f}",
        f"  Target entropy:   {comp.target_entropy:.4f}",
        f"  Delta (T-B):      {comp.delta:+.4f}",
        "",
        f"Context window (positions {comp.context_start}–{ctx_end}):",
        f"  Baseline: {[round(e, 4) for e in comp.context_baseline]}",
        f"  Target:   {[round(e, 4) for e in comp.context_target]}",
    ]
    return "\n".join(lines)


def load_logprobs_file(path: str) -> list[dict[str, float]]:
    """Load logprobs from a JSON file.

    Expected format: list of dicts, each mapping token string to logprob float.
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        msg = f"Expected JSON array, got {type(data).__name__}"
        raise ValueError(msg)
    return data
