"""Concurrency scaling analysis: measure divergence rate at different concurrency levels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from xpyd_acc.log import get_logger

logger = get_logger("concurrency_sweep")


@dataclass
class SweepLevelResult:
    """Result for a single concurrency level."""

    concurrency: int
    total_samples: int
    divergent_samples: int
    divergence_rate: float
    # Optional timing
    elapsed_seconds: float | None = None


@dataclass
class SweepResult:
    """Aggregated results across all concurrency levels."""

    levels: list[SweepLevelResult] = field(default_factory=list)
    dataset_path: str | None = None
    baseline_url: str | None = None
    target_url: str | None = None
    model: str | None = None
    any_divergence: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "levels": [asdict(lv) for lv in self.levels],
            "dataset_path": self.dataset_path,
            "baseline_url": self.baseline_url,
            "target_url": self.target_url,
            "model": self.model,
            "any_divergence": self.any_divergence,
        }

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepResult:
        """Deserialise from a dict."""
        levels = [SweepLevelResult(**lv) for lv in data.get("levels", [])]
        return cls(
            levels=levels,
            dataset_path=data.get("dataset_path"),
            baseline_url=data.get("baseline_url"),
            target_url=data.get("target_url"),
            model=data.get("model"),
            any_divergence=data.get("any_divergence", False),
        )


def format_sweep(result: SweepResult) -> str:
    """Format sweep results as a rich-friendly table string."""
    lines: list[str] = []
    lines.append("Concurrency Sweep Results")
    lines.append("=" * 60)
    lines.append(f"{'Concurrency':>12} {'Total':>8} {'Divergent':>10} {'Rate':>10}")
    lines.append("-" * 60)
    for lv in result.levels:
        rate_str = f"{lv.divergence_rate:.2%}"
        elapsed = f"  ({lv.elapsed_seconds:.1f}s)" if lv.elapsed_seconds is not None else ""
        lines.append(
            f"{lv.concurrency:>12} {lv.total_samples:>8} "
            f"{lv.divergent_samples:>10} {rate_str:>10}{elapsed}"
        )
    lines.append("-" * 60)
    status = "DIVERGENCE DETECTED" if result.any_divergence else "ALL MATCH"
    lines.append(f"Status: {status}")
    return "\n".join(lines)


async def run_sweep(
    *,
    baseline_url: str,
    target_url: str,
    dataset_path: str,
    levels: list[int],
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    template_path: str | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 120.0,
    skip_validation: bool = False,
    on_level_complete: Callable[[SweepLevelResult], None] | None = None,
) -> SweepResult:
    """Run batch comparison at multiple concurrency levels.

    Parameters
    ----------
    baseline_url : str
        Baseline endpoint URL.
    target_url : str
        Target endpoint URL.
    dataset_path : str
        Path to the dataset file.
    levels : list[int]
        Concurrency levels to test (e.g. [1, 2, 4, 8]).
    on_level_complete : callable, optional
        Callback invoked after each level finishes.

    Returns
    -------
    SweepResult
        Aggregated results across all levels.
    """
    import time

    from xpyd_acc.batch_compare import run_batch
    from xpyd_acc.sampling import SamplingParams

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    sweep = SweepResult(
        dataset_path=dataset_path,
        baseline_url=baseline_url,
        target_url=target_url,
        model=model,
    )

    for conc in sorted(levels):
        logger.info("Running concurrency level %d", conc)
        t0 = time.monotonic()

        report = await run_batch(
            baseline_url=baseline_url,
            target_url=target_url,
            dataset_path=dataset_path,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            sampling_params=sampling,
            concurrency=conc,
            template_path=template_path,
            retries=retries,
            retry_delay=retry_delay,
            timeout=timeout,
            skip_validation=skip_validation,
        )

        elapsed = time.monotonic() - t0
        lv_result = SweepLevelResult(
            concurrency=conc,
            total_samples=report.total_samples,
            divergent_samples=report.divergent_samples,
            divergence_rate=report.divergence_rate,
            elapsed_seconds=round(elapsed, 2),
        )
        sweep.levels.append(lv_result)

        if report.divergent_samples > 0:
            sweep.any_divergence = True

        if on_level_complete:
            on_level_complete(lv_result)

    return sweep
