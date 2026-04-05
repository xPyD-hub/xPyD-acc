"""Auto-bisect divergence by context length.

Binary search over prompt prefix lengths to find the minimum context
length where PD disaggregation diverges from baseline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Callable

from .logprobs import LogprobsCollector, LogprobsComparator
from .sampling import SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class BisectStep:
    """Result of a single bisect iteration."""

    iteration: int
    prefix_length: int
    diverges: bool
    first_divergence_index: int | None = None


@dataclass
class BisectResult:
    """Result of the full bisect search."""

    threshold_length: int | None
    total_iterations: int
    steps: list[BisectStep] = field(default_factory=list)
    always_diverges: bool = False
    never_diverges: bool = False

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)


async def _check_divergence(
    baseline_url: str,
    target_url: str,
    prompt: str,
    model: str,
    prefix_length: int,
    *,
    api_key: str | None = None,
    sampling: SamplingParams | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 120.0,
) -> tuple[bool, int | None]:
    """Check if the given prefix length causes divergence."""
    prefix = prompt[:prefix_length]
    if not prefix:
        return False, None

    key = api_key or "no-key"
    baseline_collector = LogprobsCollector(baseline_url, api_key=key, model=model)
    target_collector = LogprobsCollector(target_url, api_key=key, model=model)

    baseline_result = await baseline_collector.collect(
        prefix, timeout=timeout, retries=retries, retry_delay=retry_delay,
        sampling_params=sampling,
    )
    target_result = await target_collector.collect(
        prefix, timeout=timeout, retries=retries, retry_delay=retry_delay,
        sampling_params=sampling,
    )

    comparator = LogprobsComparator()
    report = comparator.compare(baseline_result, target_result)
    if report.match:
        return False, None
    div_idx = report.divergence.token_index if report.divergence else None
    return True, div_idx


async def run_bisect(
    baseline_url: str,
    target_url: str,
    prompt: str,
    model: str,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    api_key: str | None = None,
    sampling: SamplingParams | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 120.0,
    progress_callback: Callable[[int, int, bool], None] | None = None,
) -> BisectResult:
    """Binary search for the minimum prefix length that causes divergence.

    Parameters
    ----------
    baseline_url : str
        Baseline (aggregated) endpoint URL.
    target_url : str
        Target (disaggregated) endpoint URL.
    prompt : str
        Full prompt text.
    model : str
        Model name for both endpoints.
    min_length : int | None
        Minimum prefix length to test (default 1).
    max_length : int | None
        Maximum prefix length to test (default len(prompt)).

    Returns
    -------
    BisectResult
        The bisect search result with threshold_length being the
        minimum prefix length that causes divergence.
    """
    lo = min_length if min_length is not None else 1
    hi = max_length if max_length is not None else len(prompt)
    lo = max(1, lo)
    hi = min(hi, len(prompt))

    if lo > hi:
        return BisectResult(threshold_length=None, total_iterations=0, never_diverges=True)

    steps: list[BisectStep] = []
    iteration = 0

    check_kwargs = dict(
        api_key=api_key, sampling=sampling, retries=retries,
        retry_delay=retry_delay, timeout=timeout,
    )

    # Check max length first — if it doesn't diverge, nothing will.
    iteration += 1
    diverges_hi, div_idx_hi = await _check_divergence(
        baseline_url, target_url, prompt, model, hi, **check_kwargs,
    )
    steps.append(BisectStep(iteration, hi, diverges_hi, div_idx_hi))
    logger.info("Bisect step %d: length=%d diverges=%s", iteration, hi, diverges_hi)
    if progress_callback:
        progress_callback(iteration, hi, diverges_hi)

    if not diverges_hi:
        return BisectResult(
            threshold_length=None, total_iterations=iteration,
            steps=steps, never_diverges=True,
        )

    # Check min length — if it diverges, divergence starts at or below min.
    iteration += 1
    diverges_lo, div_idx_lo = await _check_divergence(
        baseline_url, target_url, prompt, model, lo, **check_kwargs,
    )
    steps.append(BisectStep(iteration, lo, diverges_lo, div_idx_lo))
    logger.info("Bisect step %d: length=%d diverges=%s", iteration, lo, diverges_lo)
    if progress_callback:
        progress_callback(iteration, lo, diverges_lo)

    if diverges_lo:
        return BisectResult(
            threshold_length=lo, total_iterations=iteration,
            steps=steps, always_diverges=True,
        )

    # Binary search: lo doesn't diverge, hi does.
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        iteration += 1
        diverges_mid, div_idx_mid = await _check_divergence(
            baseline_url, target_url, prompt, model, mid, **check_kwargs,
        )
        steps.append(BisectStep(iteration, mid, diverges_mid, div_idx_mid))
        logger.info("Bisect step %d: length=%d diverges=%s", iteration, mid, diverges_mid)
        if progress_callback:
            progress_callback(iteration, mid, diverges_mid)

        if diverges_mid:
            hi = mid
        else:
            lo = mid

    return BisectResult(
        threshold_length=hi, total_iterations=iteration, steps=steps,
    )
