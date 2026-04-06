"""Automatic threshold tuning from historical batch reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xpyd_acc.batch_compare import BatchReport, load_report
from xpyd_acc.log import get_logger

logger = get_logger("auto_threshold")


@dataclass
class ThresholdRecommendation:
    """Recommended threshold values based on historical report analysis."""

    fail_threshold: float | None
    numeric_tolerance: float | None
    confidence: str  # "high", "medium", "low"
    sample_size: int
    reasoning: list[str]
    divergence_rates: list[float] = field(default_factory=list)
    logprob_gaps: list[float] = field(default_factory=list)
    percentile_used: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "fail_threshold": self.fail_threshold,
            "numeric_tolerance": self.numeric_tolerance,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "reasoning": self.reasoning,
            "divergence_rates": self.divergence_rates,
            "logprob_gaps": self.logprob_gaps,
            "percentile_used": self.percentile_used,
        }


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list (0 < p < 1)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p
    floor_k = int(k)
    ceil_k = min(floor_k + 1, len(sorted_data) - 1)
    frac = k - floor_k
    return sorted_data[floor_k] + frac * (sorted_data[ceil_k] - sorted_data[floor_k])


def analyze_thresholds(
    reports: list[BatchReport],
    percentile_level: float = 0.95,
) -> ThresholdRecommendation:
    """Analyze batch reports and recommend threshold values.

    Args:
        reports: List of BatchReport objects to analyze.
        percentile_level: Percentile to use for recommendations (0-1).

    Returns:
        ThresholdRecommendation with suggested values.
    """
    if not reports:
        return ThresholdRecommendation(
            fail_threshold=None,
            numeric_tolerance=None,
            confidence="low",
            sample_size=0,
            reasoning=["No reports provided — cannot compute thresholds."],
            percentile_used=percentile_level,
        )

    divergence_rates: list[float] = []
    logprob_gaps: list[float] = []
    total_samples = 0

    for report in reports:
        divergence_rates.append(report.divergence_rate)
        total_samples += report.total_samples
        for result in report.results:
            if result.logprob_gap is not None and not result.exact_match:
                logprob_gaps.append(result.logprob_gap)

    reasoning: list[str] = []

    # Fail threshold recommendation
    fail_threshold: float | None = None
    if divergence_rates:
        p_val = _percentile(divergence_rates, percentile_level)
        # Round up slightly to give headroom
        fail_threshold = round(min(p_val * 1.1 + 0.005, 1.0), 3)
        reasoning.append(
            f"Observed divergence rates: min={min(divergence_rates):.3f}, "
            f"max={max(divergence_rates):.3f}, "
            f"p{int(percentile_level * 100)}={p_val:.3f} "
            f"→ recommended fail_threshold={fail_threshold:.3f}"
        )

    # Numeric tolerance recommendation
    numeric_tolerance: float | None = None
    if logprob_gaps:
        p_val = _percentile(logprob_gaps, percentile_level)
        numeric_tolerance = round(p_val, 4)
        reasoning.append(
            f"Observed logprob gaps at divergence: min={min(logprob_gaps):.4f}, "
            f"max={max(logprob_gaps):.4f}, "
            f"p{int(percentile_level * 100)}={p_val:.4f} "
            f"→ recommended numeric_tolerance={numeric_tolerance:.4f}"
        )
    else:
        reasoning.append("No logprob gap data available — cannot recommend numeric_tolerance.")

    # Confidence assessment
    if total_samples >= 500 and len(reports) >= 3:
        confidence = "high"
    elif total_samples >= 100 or len(reports) >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    reasoning.append(
        f"Analyzed {len(reports)} report(s) with {total_samples} total samples "
        f"→ confidence: {confidence}"
    )

    return ThresholdRecommendation(
        fail_threshold=fail_threshold,
        numeric_tolerance=numeric_tolerance,
        confidence=confidence,
        sample_size=total_samples,
        reasoning=reasoning,
        divergence_rates=divergence_rates,
        logprob_gaps=logprob_gaps,
        percentile_used=percentile_level,
    )


def load_reports(paths: list[str]) -> list[BatchReport]:
    """Load multiple batch reports from file paths."""
    reports: list[BatchReport] = []
    for p in paths:
        try:
            reports.append(load_report(p))
        except Exception as exc:
            logger.warning("Failed to load report %s: %s", p, exc)
    return reports


def format_recommendations(rec: ThresholdRecommendation) -> str:
    """Format recommendations as a human-readable string."""
    lines: list[str] = []
    lines.append("═══ Threshold Recommendations ═══")
    lines.append("")

    if rec.fail_threshold is not None:
        lines.append(f"  fail_threshold:    {rec.fail_threshold:.3f}")
    else:
        lines.append("  fail_threshold:    (no data)")

    if rec.numeric_tolerance is not None:
        lines.append(f"  numeric_tolerance: {rec.numeric_tolerance:.4f}")
    else:
        lines.append("  numeric_tolerance: (no data)")

    lines.append(f"  confidence:        {rec.confidence}")
    lines.append(f"  sample_size:       {rec.sample_size}")
    lines.append(f"  percentile:        p{int(rec.percentile_used * 100)}")
    lines.append("")

    if rec.divergence_rates:
        lines.append(f"  Divergence rates:  {', '.join(f'{r:.3f}' for r in rec.divergence_rates)}")
    if rec.logprob_gaps:
        n = len(rec.logprob_gaps)
        lines.append(f"  Logprob gap samples: {n}")

    lines.append("")
    lines.append("Reasoning:")
    for r in rec.reasoning:
        lines.append(f"  • {r}")

    return "\n".join(lines)
