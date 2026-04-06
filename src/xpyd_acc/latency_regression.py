"""Endpoint response time regression detection (M84)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from xpyd_acc.log import get_logger

logger = get_logger("latency_regression")


@dataclass
class LatencyRegressionResult:
    """Result of comparing two benchmark runs."""

    old_mean_ms: float
    new_mean_ms: float
    mean_diff_ms: float
    p_value: float
    cohens_d: float
    old_p50_ms: float
    new_p50_ms: float
    old_p95_ms: float
    new_p95_ms: float
    old_p99_ms: float
    new_p99_ms: float
    verdict: str  # "faster", "slower", "unchanged"
    alpha: float
    old_count: int
    new_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _welch_t_test(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
) -> tuple[float, float]:
    """Welch's t-test. Returns (t_statistic, p_value).

    Two-tailed test. Uses approximation for p-value from t-distribution.
    """
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    se1 = (std1**2) / n1
    se2 = (std2**2) / n2
    se_sum = se1 + se2

    if se_sum == 0:
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / math.sqrt(se_sum)

    # Welch-Satterthwaite degrees of freedom
    if se1 == 0 and se2 == 0:
        df = n1 + n2 - 2
    else:
        num = se_sum**2
        denom = (se1**2 / (n1 - 1)) + (se2**2 / (n2 - 1))
        if denom == 0:
            df = n1 + n2 - 2
        else:
            df = num / denom

    # Approximate two-tailed p-value using regularized incomplete beta function
    p_value = _t_distribution_p_value(abs(t_stat), df)
    return t_stat, p_value


def _t_distribution_p_value(t: float, df: float) -> float:
    """Approximate two-tailed p-value for t-distribution.

    Uses the relationship between t-distribution and regularized incomplete beta.
    """
    if df <= 0:
        return 1.0
    x = df / (df + t**2)
    p = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    return p


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the continued fraction expansion
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(
        math.log(x) * a + math.log(1 - x) * b - ln_beta
    ) / a

    # Lentz's algorithm for continued fraction
    # I_x(a,b) = front * cf where cf is a continued fraction
    # Using the standard DLMF 8.17.22 expansion
    def _cf(a: float, b: float, x: float) -> float:
        max_iter = 200
        tiny = 1e-30
        f = 1.0
        c = 1.0
        d = 1.0 - (a + b) * x / (a + 1)
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        f = d

        for m in range(1, max_iter + 1):
            # Even step
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            d = 1.0 + num * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + num / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            f *= d * c

            # Odd step
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + num / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            f *= delta

            if abs(delta - 1.0) < 1e-10:
                break

        return f

    # For numerical stability, use symmetry when x > (a+1)/(a+b+2)
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_incomplete_beta(b, a, 1 - x)

    return front * _cf(a, b, x)


def _cohens_d(
    mean1: float, std1: float, n1: int,
    mean2: float, std2: float, n2: int,
) -> float:
    """Cohen's d effect size (pooled standard deviation)."""
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return (mean2 - mean1) / pooled_std


def _percentile(data: list[float], p: float) -> float:
    """Compute percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def _load_benchmark(path: Path) -> dict:
    """Load benchmark JSON file."""
    with open(path) as f:
        return json.load(f)


def run_latency_regression(
    old_path: Path,
    new_path: Path,
    alpha: float = 0.05,
) -> LatencyRegressionResult:
    """Compare two benchmark runs for latency regression.

    Args:
        old_path: Path to old (baseline) benchmark JSON.
        new_path: Path to new (current) benchmark JSON.
        alpha: Significance level for the t-test.

    Returns:
        LatencyRegressionResult with test results and verdict.
    """
    old_data = _load_benchmark(old_path)
    new_data = _load_benchmark(new_path)

    old_latencies = old_data.get("latencies_ms", [])
    new_latencies = new_data.get("latencies_ms", [])

    if not old_latencies or not new_latencies:
        raise ValueError("Both benchmark files must contain non-empty latencies_ms")

    import statistics as stats

    old_mean = stats.mean(old_latencies)
    new_mean = stats.mean(new_latencies)
    old_std = stats.stdev(old_latencies) if len(old_latencies) > 1 else 0.0
    new_std = stats.stdev(new_latencies) if len(new_latencies) > 1 else 0.0
    n_old = len(old_latencies)
    n_new = len(new_latencies)

    _, p_value = _welch_t_test(old_mean, old_std, n_old, new_mean, new_std, n_new)
    d = _cohens_d(old_mean, old_std, n_old, new_mean, new_std, n_new)

    if p_value < alpha:
        verdict = "slower" if new_mean > old_mean else "faster"
    else:
        verdict = "unchanged"

    return LatencyRegressionResult(
        old_mean_ms=old_mean,
        new_mean_ms=new_mean,
        mean_diff_ms=new_mean - old_mean,
        p_value=p_value,
        cohens_d=d,
        old_p50_ms=_percentile(old_latencies, 50),
        new_p50_ms=_percentile(new_latencies, 50),
        old_p95_ms=_percentile(old_latencies, 95),
        new_p95_ms=_percentile(new_latencies, 95),
        old_p99_ms=_percentile(old_latencies, 99),
        new_p99_ms=_percentile(new_latencies, 99),
        verdict=verdict,
        alpha=alpha,
        old_count=n_old,
        new_count=n_new,
    )


def format_latency_regression(result: LatencyRegressionResult) -> None:
    """Print formatted latency regression report."""
    console = Console()

    # Verdict banner
    if result.verdict == "slower":
        console.print("\n[bold red]⚠️  REGRESSION DETECTED[/bold red]")
    elif result.verdict == "faster":
        console.print("\n[bold green]✅ IMPROVEMENT DETECTED[/bold green]")
    else:
        console.print("\n[bold blue]➡️  NO SIGNIFICANT CHANGE[/bold blue]")

    # Summary table
    table = Table(title="Latency Regression Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Old", justify="right")
    table.add_column("New", justify="right")
    table.add_column("Diff", justify="right")

    table.add_row(
        "Mean (ms)",
        f"{result.old_mean_ms:.2f}",
        f"{result.new_mean_ms:.2f}",
        f"{result.mean_diff_ms:+.2f}",
    )
    table.add_row(
        "P50 (ms)",
        f"{result.old_p50_ms:.2f}",
        f"{result.new_p50_ms:.2f}",
        f"{result.new_p50_ms - result.old_p50_ms:+.2f}",
    )
    table.add_row(
        "P95 (ms)",
        f"{result.old_p95_ms:.2f}",
        f"{result.new_p95_ms:.2f}",
        f"{result.new_p95_ms - result.old_p95_ms:+.2f}",
    )
    table.add_row(
        "P99 (ms)",
        f"{result.old_p99_ms:.2f}",
        f"{result.new_p99_ms:.2f}",
        f"{result.new_p99_ms - result.old_p99_ms:+.2f}",
    )
    console.print(table)

    # Statistics
    console.print(f"\n  Samples:    old={result.old_count}, new={result.new_count}")
    console.print(f"  p-value:    {result.p_value:.6f} (alpha={result.alpha})")
    console.print(f"  Cohen's d:  {result.cohens_d:.4f}")

    effect = "negligible"
    d_abs = abs(result.cohens_d)
    if d_abs >= 0.8:
        effect = "large"
    elif d_abs >= 0.5:
        effect = "medium"
    elif d_abs >= 0.2:
        effect = "small"
    console.print(f"  Effect:     {effect}")
    console.print(f"  Verdict:    {result.verdict}\n")
