"""A/B testing for comparing divergence rates between two batch reports.

Performs Fisher's exact test and chi-square test to determine whether the
difference in divergence rates between two target endpoints is statistically
significant.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ABTestResult:
    """Result of an A/B test comparing two batch reports."""

    # Report A stats
    report_a_total: int
    report_a_divergent: int
    report_a_rate: float

    # Report B stats
    report_b_total: int
    report_b_divergent: int
    report_b_rate: float

    # Rate difference
    rate_difference: float  # B - A
    rate_difference_ci_lower: float
    rate_difference_ci_upper: float

    # Statistical tests
    fisher_p_value: float
    chi_square_statistic: float
    chi_square_p_value: float
    odds_ratio: float | None  # None if undefined (zero cell)

    # Verdict
    alpha: float
    significant: bool  # True if p < alpha

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    def save_json(self, path: str | Path) -> None:
        """Write JSON to file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")


def _log_factorial(n: int) -> float:
    """Compute log(n!) using Stirling's approximation for large n."""
    if n <= 1:
        return 0.0
    return sum(math.log(i) for i in range(2, n + 1))


def _hypergeometric_log_pmf(k: int, n: int, K: int, N: int) -> float:
    """Log probability mass function of hypergeometric distribution.

    P(X=k) = C(K,k) * C(N-K, n-k) / C(N, n)
    """
    return (
        _log_comb(K, k)
        + _log_comb(N - K, n - k)
        - _log_comb(N, n)
    )


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return -math.inf
    return _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact test for a 2x2 contingency table.

    Table:
        [[a, b],
         [c, d]]

    Returns p-value.
    """
    n = a + b + c + d
    row1 = a + b
    col1 = a + c

    # Log probability of observed table
    log_p_observed = _hypergeometric_log_pmf(a, row1, col1, n)

    # Sum probabilities of all tables with P <= P(observed)
    total_p = 0.0
    min_a = max(0, row1 + col1 - n)
    max_a = min(row1, col1)

    for a_i in range(min_a, max_a + 1):
        log_p_i = _hypergeometric_log_pmf(a_i, row1, col1, n)
        if log_p_i <= log_p_observed + 1e-10:  # small tolerance for floating point
            total_p += math.exp(log_p_i)

    return min(total_p, 1.0)


def chi_square_test(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Chi-square test with Yates' correction for a 2x2 table.

    Returns (chi_square_statistic, p_value).
    """
    n = a + b + c + d
    if n == 0:
        return 0.0, 1.0

    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d

    # Check for zero marginals
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 0.0, 1.0

    # Yates' correction
    numerator = (abs(a * d - b * c) - n / 2) ** 2 * n
    denominator = row1 * row2 * col1 * col2

    if denominator == 0:
        return 0.0, 1.0

    chi2 = numerator / denominator
    chi2 = max(chi2, 0.0)

    # p-value from chi-square distribution with 1 df
    p_value = _chi2_sf(chi2, 1)
    return chi2, p_value


def _chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of chi-square distribution.

    Uses the regularized incomplete gamma function for df=1.
    """
    if x <= 0:
        return 1.0
    if df == 1:
        # For df=1: P(X > x) = 2 * (1 - Phi(sqrt(x))) = erfc(sqrt(x/2))
        return math.erfc(math.sqrt(x / 2))
    # General case via incomplete gamma (not needed for our use case)
    return math.erfc(math.sqrt(x / 2))


def _rate_difference_ci(
    n1: int, x1: int, n2: int, x2: int, confidence: float = 0.95,
) -> tuple[float, float]:
    """Wald confidence interval for the difference of two proportions (p2 - p1)."""
    if n1 == 0 or n2 == 0:
        return -1.0, 1.0

    p1 = x1 / n1
    p2 = x2 / n2
    diff = p2 - p1

    z = _normal_quantile((1 + confidence) / 2)
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    lower = max(diff - z * se, -1.0)
    upper = min(diff + z * se, 1.0)
    return lower, upper


def _normal_quantile(p: float) -> float:
    """Approximate quantile of the standard normal distribution.

    Uses Abramowitz and Stegun approximation 26.2.23.
    """
    if p <= 0 or p >= 1:
        return 0.0
    if p < 0.5:
        return -_normal_quantile(1 - p)
    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


def run_ab_test(
    report_a_total: int,
    report_a_divergent: int,
    report_b_total: int,
    report_b_divergent: int,
    alpha: float = 0.05,
) -> ABTestResult:
    """Run A/B test comparing divergence rates from two batch reports.

    Args:
        report_a_total: Total samples in report A.
        report_a_divergent: Divergent samples in report A.
        report_b_total: Total samples in report B.
        report_b_divergent: Divergent samples in report B.
        alpha: Significance level (default 0.05).

    Returns:
        ABTestResult with test statistics and verdict.
    """
    rate_a = report_a_divergent / report_a_total if report_a_total > 0 else 0.0
    rate_b = report_b_divergent / report_b_total if report_b_total > 0 else 0.0

    # 2x2 contingency table:
    # [[a_match, a_div], [b_match, b_div]]
    a_match = report_a_total - report_a_divergent
    a_div = report_a_divergent
    b_match = report_b_total - report_b_divergent
    b_div = report_b_divergent

    # Fisher's exact test
    fisher_p = fisher_exact_two_sided(a_match, a_div, b_match, b_div)

    # Chi-square test
    chi2_stat, chi2_p = chi_square_test(a_match, a_div, b_match, b_div)

    # Odds ratio
    if a_div > 0 and b_match > 0:
        odds_ratio: float | None = (a_match * b_div) / (a_div * b_match)
    elif a_div == 0 and b_div == 0:
        odds_ratio = 1.0
    else:
        odds_ratio = None

    # CI for rate difference
    ci_lower, ci_upper = _rate_difference_ci(
        report_a_total, report_a_divergent,
        report_b_total, report_b_divergent,
    )

    # Use Fisher p-value for significance decision
    significant = fisher_p < alpha

    return ABTestResult(
        report_a_total=report_a_total,
        report_a_divergent=report_a_divergent,
        report_a_rate=round(rate_a, 6),
        report_b_total=report_b_total,
        report_b_divergent=report_b_divergent,
        report_b_rate=round(rate_b, 6),
        rate_difference=round(rate_b - rate_a, 6),
        rate_difference_ci_lower=round(ci_lower, 6),
        rate_difference_ci_upper=round(ci_upper, 6),
        fisher_p_value=round(fisher_p, 6),
        chi_square_statistic=round(chi2_stat, 6),
        chi_square_p_value=round(chi2_p, 6),
        odds_ratio=round(odds_ratio, 6) if odds_ratio is not None else None,
        alpha=alpha,
        significant=significant,
    )


def format_ab_test(result: ABTestResult) -> str:
    """Format A/B test result as rich terminal text."""
    lines = [
        "=== A/B Test: Divergence Rate Comparison ===",
        "",
        f"  Report A: {result.report_a_divergent}/{result.report_a_total} "
        f"divergent ({result.report_a_rate:.2%})",
        f"  Report B: {result.report_b_divergent}/{result.report_b_total} "
        f"divergent ({result.report_b_rate:.2%})",
        "",
        f"  Rate difference (B - A): {result.rate_difference:+.4f} "
        f"[{result.rate_difference_ci_lower:+.4f}, {result.rate_difference_ci_upper:+.4f}]",
    ]
    if result.odds_ratio is not None:
        lines.append(f"  Odds ratio: {result.odds_ratio:.4f}")
    else:
        lines.append("  Odds ratio: undefined (zero cell)")
    lines.extend([
        "",
        f"  Fisher's exact p-value: {result.fisher_p_value:.6f}",
        f"  Chi-square statistic:   {result.chi_square_statistic:.4f}  "
        f"(p = {result.chi_square_p_value:.6f})",
        "",
        f"  Alpha: {result.alpha}",
    ])
    if result.significant:
        lines.append("  ❌ SIGNIFICANT DIFFERENCE — divergence rates differ (p < alpha)")
    else:
        lines.append("  ✅ NO SIGNIFICANT DIFFERENCE — divergence rates are compatible (p ≥ alpha)")

    return "\n".join(lines)
