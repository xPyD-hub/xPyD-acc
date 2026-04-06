"""Smart retry for divergent samples — classify as deterministic vs stochastic."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .batch_compare import (
    BatchReport,
    DatasetSample,
    SampleResult,
    run_batch,
)
from .log import get_logger

logger = get_logger(__name__)


@dataclass
class SampleRetryResult:
    """Result of retrying a single divergent sample with greedy settings."""

    sample_id: str
    original_classification: str
    retry_match: bool
    retry_classification: str  # "deterministic" or "stochastic"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


@dataclass
class SmartRetryResult:
    """Aggregate result of smart retry for all divergent samples."""

    original_divergent: int
    deterministic_count: int
    stochastic_count: int
    deterministic_rate: float
    stochastic_rate: float
    per_sample: list[SampleRetryResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "original_divergent": self.original_divergent,
            "deterministic_count": self.deterministic_count,
            "stochastic_count": self.stochastic_count,
            "deterministic_rate": self.deterministic_rate,
            "stochastic_rate": self.stochastic_rate,
            "per_sample": [s.to_dict() for s in self.per_sample],
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


async def run_smart_retry(
    report: BatchReport,
    baseline_url: str,
    target_url: str,
    *,
    model: str = "default",
    max_tokens: int = 64,
    api_key: str = "no-key",
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 120.0,
    skip_validation: bool = False,
    custom_headers: dict[str, str] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> SmartRetryResult:
    """Retry divergent samples with deterministic settings (temperature=0, seed=42).

    Classifies each divergence as:
    - ``deterministic``: still diverges under greedy decoding → likely a real bug
    - ``stochastic``: matches under greedy decoding → likely sampling noise

    Args:
        report: The original batch report containing divergent results.
        baseline_url: Baseline endpoint URL.
        target_url: Target endpoint URL.
        model: Model identifier.
        max_tokens: Maximum tokens per request.
        api_key: API key for authentication.
        retries: Number of HTTP retries.
        retry_delay: Base delay between retries.
        timeout: HTTP request timeout in seconds.
        skip_validation: Skip response schema validation.
        custom_headers: Optional custom HTTP headers.
        on_progress: Progress callback (completed, total).

    Returns:
        SmartRetryResult with per-sample classification.
    """
    divergent = [r for r in report.results if r.is_divergent()]
    if not divergent:
        return SmartRetryResult(
            original_divergent=0,
            deterministic_count=0,
            stochastic_count=0,
            deterministic_rate=0.0,
            stochastic_rate=0.0,
        )

    # Build dataset samples from divergent results
    samples = [
        DatasetSample(id=r.sample_id, prompt=r.prompt)
        for r in divergent
    ]

    logger.info(
        "Smart retry: re-running %d divergent samples with greedy settings",
        len(samples),
    )

    # Use a simple SamplingParams-like object for greedy decoding
    from dataclasses import dataclass as _dc

    @_dc
    class _GreedyParams:
        temperature: float = 0.0
        top_p: float | None = None
        seed: int = 42

    retry_report = await run_batch(
        samples,
        baseline_url,
        target_url,
        model=model,
        max_tokens=max_tokens,
        api_key=api_key,
        retries=retries,
        retry_delay=retry_delay,
        timeout=timeout,
        skip_validation=skip_validation,
        custom_headers=custom_headers,
        on_progress=on_progress,
        sampling_params=_GreedyParams(),
        concurrency=5,
    )

    # Build result map
    retry_map: dict[str, SampleResult] = {
        r.sample_id: r for r in retry_report.results
    }

    per_sample: list[SampleRetryResult] = []
    deterministic = 0
    stochastic = 0

    for orig in divergent:
        retry_result = retry_map.get(orig.sample_id)
        if retry_result is None:
            # Should not happen, but treat as deterministic (safe default)
            classification = "deterministic"
            retry_match = False
        else:
            retry_match = retry_result.exact_match
            classification = "stochastic" if retry_match else "deterministic"

        if classification == "deterministic":
            deterministic += 1
        else:
            stochastic += 1

        per_sample.append(
            SampleRetryResult(
                sample_id=orig.sample_id,
                original_classification=orig.classification,
                retry_match=retry_match,
                retry_classification=classification,
            )
        )

    total = len(divergent)
    return SmartRetryResult(
        original_divergent=total,
        deterministic_count=deterministic,
        stochastic_count=stochastic,
        deterministic_rate=deterministic / total if total else 0.0,
        stochastic_rate=stochastic / total if total else 0.0,
        per_sample=per_sample,
    )


def format_smart_retry(result: SmartRetryResult) -> str:
    """Format smart retry result for terminal display."""
    lines = [
        "Smart Retry Results",
        "=" * 40,
        f"Original divergent samples: {result.original_divergent}",
        f"Deterministic (real bugs):   {result.deterministic_count}"
        f" ({result.deterministic_rate:.1%})",
        f"Stochastic (sampling noise): {result.stochastic_count}"
        f" ({result.stochastic_rate:.1%})",
        "",
    ]

    if result.per_sample:
        lines.append("Per-Sample Breakdown:")
        lines.append(f"{'Sample ID':<30} {'Original':<20} {'Retry':<15}")
        lines.append("-" * 65)
        for s in result.per_sample:
            lines.append(
                f"{s.sample_id:<30} {s.original_classification:<20} "
                f"{s.retry_classification:<15}"
            )

    return "\n".join(lines)
