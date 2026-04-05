"""Output length bias detection for batch comparison reports."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampleLength:
    """Length data for a single sample."""

    sample_id: str
    baseline_length: int
    target_length: int
    length_diff: int  # target - baseline
    length_ratio: float  # target / baseline (0.0 if baseline is 0)


@dataclass
class LengthBiasResult:
    """Aggregate length bias analysis result."""

    sample_count: int
    mean_baseline_length: float
    mean_target_length: float
    mean_diff: float
    median_diff: float
    stdev_diff: float
    t_statistic: float
    p_value: float
    classification: str  # "shorter_bias", "longer_bias", "no_bias"
    alpha: float
    samples: list[SampleLength] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mean_baseline_length": self.mean_baseline_length,
            "mean_target_length": self.mean_target_length,
            "mean_diff": self.mean_diff,
            "median_diff": self.median_diff,
            "stdev_diff": self.stdev_diff,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "classification": self.classification,
            "alpha": self.alpha,
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "baseline_length": s.baseline_length,
                    "target_length": s.target_length,
                    "length_diff": s.length_diff,
                    "length_ratio": s.length_ratio,
                }
                for s in self.samples
            ],
        }


def _paired_t_test(diffs: list[int]) -> tuple[float, float]:
    """Paired t-test for mean difference from zero.

    Returns (t_statistic, p_value). Uses two-tailed test.
    For n < 2, returns (0.0, 1.0).
    """
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0

    mean_d = statistics.mean(diffs)
    stdev_d = statistics.stdev(diffs)
    if stdev_d == 0:
        # All diffs identical — if mean is 0, no bias; if non-zero, p → 0
        if mean_d == 0:
            return 0.0, 1.0
        return float("inf") if mean_d > 0 else float("-inf"), 0.0

    t_stat = mean_d / (stdev_d / math.sqrt(n))
    # Approximate p-value using normal distribution for large n
    # For small n this is an approximation (proper would use t-distribution)
    p_value = 2.0 * _normal_cdf(-abs(t_stat))
    return t_stat, p_value


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def analyze_length_bias(
    report: dict[str, Any],
    alpha: float = 0.05,
) -> LengthBiasResult:
    """Analyze output length bias from a batch comparison report.

    Parameters
    ----------
    report : dict
        Deserialized batch comparison report JSON.
    alpha : float
        Significance level for the t-test.

    Returns
    -------
    LengthBiasResult
        Analysis results with per-sample data and classification.
    """
    results = report.get("results", [])
    samples: list[SampleLength] = []

    for r in results:
        bl_out = r.get("baseline_output", "") or ""
        tg_out = r.get("target_output", "") or ""
        bl_len = len(bl_out)
        tg_len = len(tg_out)
        diff = tg_len - bl_len
        ratio = tg_len / bl_len if bl_len > 0 else 0.0
        samples.append(
            SampleLength(
                sample_id=r.get("sample_id", ""),
                baseline_length=bl_len,
                target_length=tg_len,
                length_diff=diff,
                length_ratio=ratio,
            )
        )

    if not samples:
        return LengthBiasResult(
            sample_count=0,
            mean_baseline_length=0.0,
            mean_target_length=0.0,
            mean_diff=0.0,
            median_diff=0.0,
            stdev_diff=0.0,
            t_statistic=0.0,
            p_value=1.0,
            classification="no_bias",
            alpha=alpha,
            samples=[],
        )

    bl_lengths = [s.baseline_length for s in samples]
    tg_lengths = [s.target_length for s in samples]
    diffs = [s.length_diff for s in samples]

    mean_bl = statistics.mean(bl_lengths)
    mean_tg = statistics.mean(tg_lengths)
    mean_diff = statistics.mean(diffs)
    median_diff = statistics.median(diffs)
    stdev_diff = statistics.stdev(diffs) if len(diffs) >= 2 else 0.0

    t_stat, p_val = _paired_t_test(diffs)

    if p_val < alpha:
        classification = "shorter_bias" if mean_diff < 0 else "longer_bias"
    else:
        classification = "no_bias"

    return LengthBiasResult(
        sample_count=len(samples),
        mean_baseline_length=mean_bl,
        mean_target_length=mean_tg,
        mean_diff=mean_diff,
        median_diff=median_diff,
        stdev_diff=stdev_diff,
        t_statistic=t_stat,
        p_value=p_val,
        classification=classification,
        alpha=alpha,
        samples=samples,
    )


def format_length_bias(result: LengthBiasResult) -> str:
    """Format length bias result for terminal display."""
    lines = [
        "Output Length Bias Analysis",
        "=" * 50,
        f"  Samples:              {result.sample_count}",
        f"  Mean baseline length: {result.mean_baseline_length:.1f}",
        f"  Mean target length:   {result.mean_target_length:.1f}",
        f"  Mean diff (T-B):      {result.mean_diff:+.1f}",
        f"  Median diff:          {result.median_diff:+.1f}",
        f"  Stdev diff:           {result.stdev_diff:.1f}",
        f"  t-statistic:          {result.t_statistic:.4f}",
        f"  p-value:              {result.p_value:.6f}",
        f"  Alpha:                {result.alpha}",
        "",
    ]

    if result.classification == "no_bias":
        lines.append("  ✅ No significant length bias detected")
    elif result.classification == "shorter_bias":
        lines.append("  ⚠️  Significant SHORTER bias in target outputs")
    else:
        lines.append("  ⚠️  Significant LONGER bias in target outputs")

    # Distribution summary
    if result.samples:
        shorter = sum(1 for s in result.samples if s.length_diff < 0)
        equal = sum(1 for s in result.samples if s.length_diff == 0)
        longer = sum(1 for s in result.samples if s.length_diff > 0)
        lines.append("")
        lines.append("  Distribution:")
        lines.append(f"    Shorter: {shorter} ({100*shorter/len(result.samples):.1f}%)")
        lines.append(f"    Equal:   {equal} ({100*equal/len(result.samples):.1f}%)")
        lines.append(f"    Longer:  {longer} ({100*longer/len(result.samples):.1f}%)")

    return "\n".join(lines)


def load_report_file(path: str) -> dict[str, Any]:
    """Load a batch report JSON file."""
    with open(path) as f:
        return json.load(f)
