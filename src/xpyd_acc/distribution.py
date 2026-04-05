"""Top-K logprob distribution analysis: KL divergence, JS divergence, overlap."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TokenDistribution:
    """Top-K token probability distribution at a single position."""

    index: int
    tokens: dict[str, float]  # token -> logprob


@dataclass
class PositionMetrics:
    """Distribution comparison metrics for one token position."""

    index: int
    kl_divergence: float
    js_divergence: float
    top_k_overlap: float  # Jaccard similarity of top-K token sets
    baseline_top: str
    target_top: str
    flagged: bool = False  # True if KL exceeds threshold


@dataclass
class DistributionReport:
    """Full distribution comparison report."""

    baseline_endpoint: str
    target_endpoint: str
    positions: list[PositionMetrics] = field(default_factory=list)
    mean_kl: float = 0.0
    max_kl: float = 0.0
    mean_js: float = 0.0
    mean_overlap: float = 0.0
    flagged_count: int = 0
    total_positions: int = 0


def _logprobs_to_probs(logprobs: dict[str, float]) -> dict[str, float]:
    """Convert logprobs to probabilities."""
    return {t: math.exp(lp) for t, lp in logprobs.items()}


def kl_divergence(
    p_logprobs: dict[str, float],
    q_logprobs: dict[str, float],
    epsilon: float = 1e-10,
) -> float:
    """Compute KL(P || Q) over the union of token sets.

    Args:
        p_logprobs: baseline token -> logprob mapping
        q_logprobs: target token -> logprob mapping
        epsilon: smoothing constant for missing tokens

    Returns:
        KL divergence (non-negative float, 0 means identical).
    """
    p_probs = _logprobs_to_probs(p_logprobs)
    q_probs = _logprobs_to_probs(q_logprobs)
    all_tokens = set(p_probs) | set(q_probs)

    kl = 0.0
    for t in all_tokens:
        p_val = p_probs.get(t, epsilon)
        q_val = q_probs.get(t, epsilon)
        if p_val > 0:
            kl += p_val * math.log(p_val / q_val)
    return max(kl, 0.0)


def js_divergence(
    p_logprobs: dict[str, float],
    q_logprobs: dict[str, float],
    epsilon: float = 1e-10,
) -> float:
    """Compute Jensen-Shannon divergence (symmetric, bounded [0, ln2]).

    Args:
        p_logprobs: baseline token -> logprob mapping
        q_logprobs: target token -> logprob mapping
        epsilon: smoothing constant for missing tokens

    Returns:
        JS divergence (non-negative float).
    """
    p_probs = _logprobs_to_probs(p_logprobs)
    q_probs = _logprobs_to_probs(q_logprobs)
    all_tokens = set(p_probs) | set(q_probs)

    # Build M = (P + Q) / 2
    m_probs: dict[str, float] = {}
    for t in all_tokens:
        m_probs[t] = (p_probs.get(t, epsilon) + q_probs.get(t, epsilon)) / 2.0

    # Convert M back to logprobs for reuse
    m_logprobs = {t: math.log(v) for t, v in m_probs.items()}

    return (kl_divergence(p_logprobs, m_logprobs, epsilon)
            + kl_divergence(q_logprobs, m_logprobs, epsilon)) / 2.0


def top_k_overlap(
    p_logprobs: dict[str, float],
    q_logprobs: dict[str, float],
) -> float:
    """Jaccard similarity of top-K token sets.

    Returns:
        Float in [0, 1]. 1.0 means identical token sets.
    """
    p_set = set(p_logprobs.keys())
    q_set = set(q_logprobs.keys())
    if not p_set and not q_set:
        return 1.0
    intersection = p_set & q_set
    union = p_set | q_set
    return len(intersection) / len(union)


def compare_distributions(
    baseline: list[TokenDistribution],
    target: list[TokenDistribution],
    kl_threshold: float = 0.1,
    baseline_endpoint: str = "",
    target_endpoint: str = "",
) -> DistributionReport:
    """Compare top-K distributions position by position.

    Args:
        baseline: per-position distributions from baseline endpoint
        target: per-position distributions from target endpoint
        kl_threshold: flag positions where KL divergence exceeds this
        baseline_endpoint: endpoint URL for report
        target_endpoint: endpoint URL for report

    Returns:
        DistributionReport with per-position and aggregate metrics.
    """
    min_len = min(len(baseline), len(target))
    positions: list[PositionMetrics] = []
    kl_values: list[float] = []
    js_values: list[float] = []
    overlap_values: list[float] = []
    flagged = 0

    for i in range(min_len):
        b = baseline[i]
        t = target[i]
        kl = kl_divergence(b.tokens, t.tokens)
        js = js_divergence(b.tokens, t.tokens)
        overlap = top_k_overlap(b.tokens, t.tokens)
        is_flagged = kl > kl_threshold

        # Find top token for each
        b_top = max(b.tokens, key=b.tokens.get) if b.tokens else ""  # type: ignore[arg-type]
        t_top = max(t.tokens, key=t.tokens.get) if t.tokens else ""  # type: ignore[arg-type]

        positions.append(PositionMetrics(
            index=i,
            kl_divergence=kl,
            js_divergence=js,
            top_k_overlap=overlap,
            baseline_top=b_top,
            target_top=t_top,
            flagged=is_flagged,
        ))
        kl_values.append(kl)
        js_values.append(js)
        overlap_values.append(overlap)
        if is_flagged:
            flagged += 1

    mean_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0
    max_kl = max(kl_values) if kl_values else 0.0
    mean_js = sum(js_values) / len(js_values) if js_values else 0.0
    mean_overlap = sum(overlap_values) / len(overlap_values) if overlap_values else 0.0

    return DistributionReport(
        baseline_endpoint=baseline_endpoint,
        target_endpoint=target_endpoint,
        positions=positions,
        mean_kl=mean_kl,
        max_kl=max_kl,
        mean_js=mean_js,
        mean_overlap=mean_overlap,
        flagged_count=flagged,
        total_positions=min_len,
    )


def format_distribution_report(report: DistributionReport) -> str:
    """Format distribution report as human-readable text."""
    lines = [
        "=== Distribution Analysis Report ===",
        f"Baseline: {report.baseline_endpoint}",
        f"Target:   {report.target_endpoint}",
        f"Positions analyzed: {report.total_positions}",
        f"Mean KL divergence: {report.mean_kl:.6f}",
        f"Max KL divergence:  {report.max_kl:.6f}",
        f"Mean JS divergence: {report.mean_js:.6f}",
        f"Mean top-K overlap: {report.mean_overlap:.4f}",
        f"Flagged positions:  {report.flagged_count}",
        "",
    ]

    flagged_positions = [p for p in report.positions if p.flagged]
    if flagged_positions:
        lines.append("Flagged positions (KL exceeds threshold):")
        for p in flagged_positions:
            lines.append(
                f"  [{p.index}] KL={p.kl_divergence:.6f} JS={p.js_divergence:.6f} "
                f"overlap={p.top_k_overlap:.4f} "
                f"baseline_top={p.baseline_top!r} target_top={p.target_top!r}"
            )
    else:
        lines.append("✅ No positions flagged — distributions are similar")

    return "\n".join(lines)
