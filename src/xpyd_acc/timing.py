"""Token timing analysis for comparing TTFT and inter-token latency between endpoints."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from xpyd_acc.streaming import StreamToken


@dataclass
class TimingStats:
    """Latency statistics for a single endpoint's streaming response."""

    ttft: float  # time to first token (seconds)
    total_tokens: int
    total_duration: float  # wall-clock seconds from start to last token
    itl_values: list[float] = field(default_factory=list)  # inter-token latencies

    @property
    def itl_p50(self) -> float | None:
        """Median inter-token latency."""
        if not self.itl_values:
            return None
        return _percentile(self.itl_values, 50)

    @property
    def itl_p95(self) -> float | None:
        """95th percentile inter-token latency."""
        if not self.itl_values:
            return None
        return _percentile(self.itl_values, 95)

    @property
    def itl_p99(self) -> float | None:
        """99th percentile inter-token latency."""
        if not self.itl_values:
            return None
        return _percentile(self.itl_values, 99)

    @property
    def itl_mean(self) -> float | None:
        """Mean inter-token latency."""
        if not self.itl_values:
            return None
        return statistics.mean(self.itl_values)

    @property
    def itl_min(self) -> float | None:
        """Minimum inter-token latency."""
        if not self.itl_values:
            return None
        return min(self.itl_values)

    @property
    def itl_max(self) -> float | None:
        """Maximum inter-token latency."""
        if not self.itl_values:
            return None
        return max(self.itl_values)


@dataclass
class TimingComparisonReport:
    """Comparison of timing between baseline and target endpoints."""

    baseline: TimingStats
    target: TimingStats
    ttft_diff: float  # target.ttft - baseline.ttft (positive = target slower)
    ttft_ratio: float  # target.ttft / baseline.ttft


def _percentile(data: list[float], pct: float) -> float:
    """Compute percentile using linear interpolation."""
    if not data:
        raise ValueError("Cannot compute percentile of empty list")
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    # Linear interpolation
    k = (pct / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def compute_timing_stats(
    tokens: list[StreamToken],
    request_start: float,
) -> TimingStats:
    """Compute timing statistics from a list of StreamTokens.

    Args:
        tokens: Tokens with monotonic timestamps.
        request_start: Monotonic time when the request was initiated.

    Returns:
        TimingStats with TTFT and inter-token latency stats.
    """
    if not tokens:
        return TimingStats(
            ttft=0.0,
            total_tokens=0,
            total_duration=0.0,
            itl_values=[],
        )

    ttft = tokens[0].timestamp - request_start
    total_duration = tokens[-1].timestamp - request_start

    itl_values: list[float] = []
    for i in range(1, len(tokens)):
        itl = tokens[i].timestamp - tokens[i - 1].timestamp
        itl_values.append(itl)

    return TimingStats(
        ttft=ttft,
        total_tokens=len(tokens),
        total_duration=total_duration,
        itl_values=itl_values,
    )


def compare_timing(
    baseline: TimingStats,
    target: TimingStats,
) -> TimingComparisonReport:
    """Compare timing stats between baseline and target.

    Args:
        baseline: Timing stats for the baseline endpoint.
        target: Timing stats for the target endpoint.

    Returns:
        TimingComparisonReport with comparison metrics.
    """
    ttft_diff = target.ttft - baseline.ttft
    ttft_ratio = target.ttft / baseline.ttft if baseline.ttft > 0 else float("inf")

    return TimingComparisonReport(
        baseline=baseline,
        target=target,
        ttft_diff=ttft_diff,
        ttft_ratio=ttft_ratio,
    )


def format_timing_report(report: TimingComparisonReport) -> str:
    """Format a timing comparison report as human-readable text."""
    lines = [
        "=== Token Timing Analysis ===",
        "",
        "--- TTFT (Time to First Token) ---",
        f"  Baseline: {report.baseline.ttft * 1000:.1f} ms",
        f"  Target:   {report.target.ttft * 1000:.1f} ms",
        f"  Diff:     {report.ttft_diff * 1000:+.1f} ms ({report.ttft_ratio:.2f}x)",
        "",
    ]

    for label, stats in [("Baseline", report.baseline), ("Target", report.target)]:
        lines.append(f"--- {label} Inter-Token Latency ---")
        lines.append(f"  Total tokens: {stats.total_tokens}")
        lines.append(f"  Duration:     {stats.total_duration * 1000:.1f} ms")
        if stats.itl_values:
            lines.append(f"  ITL mean:     {stats.itl_mean * 1000:.2f} ms")
            lines.append(f"  ITL p50:      {stats.itl_p50 * 1000:.2f} ms")
            lines.append(f"  ITL p95:      {stats.itl_p95 * 1000:.2f} ms")
            lines.append(f"  ITL p99:      {stats.itl_p99 * 1000:.2f} ms")
            lines.append(f"  ITL min:      {stats.itl_min * 1000:.2f} ms")
            lines.append(f"  ITL max:      {stats.itl_max * 1000:.2f} ms")
        else:
            lines.append("  ITL:          N/A (≤1 token)")
        lines.append("")

    return "\n".join(lines)
