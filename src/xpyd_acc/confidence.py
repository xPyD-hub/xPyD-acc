"""Wilson score confidence intervals for binomial proportions."""

from __future__ import annotations

import math


def wilson_ci(
    divergent: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    Args:
        divergent: Number of divergent (positive) samples.
        total: Total number of samples.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound) as floats in [0, 1].

    Raises:
        ValueError: If inputs are invalid.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if divergent < 0 or divergent > total:
        raise ValueError(
            f"divergent must be in [0, total], got {divergent}/{total}"
        )
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    # Z-score from normal distribution
    z = _normal_ppf((1 + confidence) / 2)
    z2 = z * z
    n = total
    p_hat = divergent / n

    denominator = 1 + z2 / n
    center = p_hat + z2 / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n)

    lower = (center - spread) / denominator
    upper = (center + spread) / denominator

    # Clamp to [0, 1]
    return max(0.0, lower), min(1.0, upper)


def _normal_ppf(p: float) -> float:
    """Approximate inverse normal CDF (percent point function).

    Uses the rational approximation from Abramowitz and Stegun.
    Accurate to ~4.5e-4 absolute error.
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0, 1), got {p}")

    if p < 0.5:
        return -_rational_approx(math.sqrt(-2.0 * math.log(p)))
    else:
        return _rational_approx(math.sqrt(-2.0 * math.log(1.0 - p)))


def _rational_approx(t: float) -> float:
    """Rational approximation for inverse normal CDF."""
    # Coefficients from Peter Acklam's approximation
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
