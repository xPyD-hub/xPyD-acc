"""Parallel multi-dataset batch comparison.

Run multiple evaluation datasets concurrently in a single command,
producing per-dataset reports and an aggregate summary.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable

from xpyd_acc.batch_compare import BatchReport, DatasetSample, run_batch
from xpyd_acc.log import get_logger

logger = get_logger("multi_dataset")


@dataclass
class MultiDatasetReport:
    """Report combining results from multiple datasets."""

    baseline_url: str
    target_url: str
    model: str
    datasets: list[str]
    per_dataset: dict[str, BatchReport]
    total_samples: int = 0
    total_divergent: int = 0
    overall_divergence_rate: float = 0.0

    def __post_init__(self) -> None:
        """Compute aggregate stats from per-dataset reports."""
        self._compute_aggregates()

    def _compute_aggregates(self) -> None:
        self.total_samples = sum(r.total_samples for r in self.per_dataset.values())
        self.total_divergent = sum(r.divergent_samples for r in self.per_dataset.values())
        self.overall_divergence_rate = (
            self.total_divergent / self.total_samples if self.total_samples > 0 else 0.0
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data: dict[str, Any] = {
            "baseline_url": self.baseline_url,
            "target_url": self.target_url,
            "model": self.model,
            "datasets": self.datasets,
            "total_samples": self.total_samples,
            "total_divergent": self.total_divergent,
            "overall_divergence_rate": self.overall_divergence_rate,
            "per_dataset": {
                name: json.loads(report.to_json())
                for name, report in self.per_dataset.items()
            },
        }
        return json.dumps(data, indent=2)

    def to_markdown(self, *, max_divergent_samples: int = 10) -> str:
        """Serialize to Markdown string."""
        lines: list[str] = []
        lines.append("# Multi-Dataset Batch Comparison Report")
        lines.append("")
        lines.append(f"**Baseline:** `{self.baseline_url}`")
        lines.append(f"**Target:** `{self.target_url}`")
        lines.append(f"**Model:** `{self.model}`")
        lines.append(f"**Datasets:** {len(self.datasets)}")
        lines.append(f"**Total samples:** {self.total_samples}")
        lines.append(
            f"**Overall divergence rate:** {self.overall_divergence_rate:.1%}"
            f" ({self.total_divergent}/{self.total_samples})"
        )
        lines.append("")

        # Per-dataset summary table
        lines.append("## Per-Dataset Summary")
        lines.append("")
        lines.append("| Dataset | Samples | Matches | Divergent | Rate |")
        lines.append("|---------|---------|---------|-----------|------|")
        for name in self.datasets:
            r = self.per_dataset[name]
            lines.append(
                f"| {name} | {r.total_samples} | {r.match_samples} "
                f"| {r.divergent_samples} | {r.divergence_rate:.1%} |"
            )
        lines.append("")

        # Per-dataset details
        for name in self.datasets:
            r = self.per_dataset[name]
            lines.append(f"## Dataset: {name}")
            lines.append("")
            divergent = [s for s in r.results if not s.exact_match]
            shown = divergent[:max_divergent_samples]
            if shown:
                lines.append("### Top Divergent Samples")
                lines.append("")
                for s in shown:
                    idx = s.first_divergence_index
                    lines.append(f"- **{s.sample_id}**: divergence at index {idx}")
                lines.append("")
            if len(divergent) > max_divergent_samples:
                lines.append(f"*... and {len(divergent) - max_divergent_samples} more*")
                lines.append("")

        return "\n".join(lines)


def format_multi_dataset_report(report: MultiDatasetReport) -> str:
    """Format a multi-dataset report for terminal display."""
    lines: list[str] = []
    lines.append("Multi-Dataset Batch Comparison Report")
    lines.append("=" * 40)
    lines.append(f"Baseline: {report.baseline_url}")
    lines.append(f"Target:   {report.target_url}")
    lines.append(f"Model:    {report.model}")
    lines.append(f"Datasets: {len(report.datasets)}")
    lines.append(
        f"Overall:  {report.overall_divergence_rate:.1%} divergence"
        f" ({report.total_divergent}/{report.total_samples})"
    )
    lines.append("")

    for name in report.datasets:
        r = report.per_dataset[name]
        status = "✓" if r.divergent_samples == 0 else "✗"
        lines.append(
            f"  {status} {name}: {r.divergence_rate:.1%}"
            f" ({r.divergent_samples}/{r.total_samples} divergent)"
        )

    return "\n".join(lines)


async def run_multi_dataset(
    dataset_map: dict[str, list[DatasetSample]],
    baseline_url: str,
    target_url: str,
    *,
    model: str = "default",
    max_tokens: int = 64,
    api_key: str = "no-key",
    logprob_gap_threshold: float = 0.1,
    concurrency: int = 5,
    retries: int = 3,
    retry_delay: float = 1.0,
    on_dataset_complete: Callable[[str, BatchReport], None] | None = None,
    match_config: Any | None = None,
    sampling_params: Any | None = None,
    timeout: float = 120.0,
    skip_validation: bool = False,
    custom_headers: dict[str, str] | None = None,
) -> MultiDatasetReport:
    """Run batch comparison for multiple datasets concurrently.

    Args:
        dataset_map: Mapping of dataset name -> list of samples.
        on_dataset_complete: Optional callback after each dataset completes.
    """
    dataset_names = list(dataset_map.keys())

    async def _run_one(name: str) -> tuple[str, BatchReport]:
        logger.info("Running batch for dataset: %s (%d samples)", name, len(dataset_map[name]))
        report = await run_batch(
            dataset_map[name],
            baseline_url,
            target_url,
            model=model,
            max_tokens=max_tokens,
            api_key=api_key,
            logprob_gap_threshold=logprob_gap_threshold,
            concurrency=concurrency,
            retries=retries,
            retry_delay=retry_delay,
            match_config=match_config,
            sampling_params=sampling_params,
            timeout=timeout,
            skip_validation=skip_validation,
            custom_headers=custom_headers,
        )
        logger.info("Dataset %s complete: %s divergence", name, f"{report.divergence_rate:.1%}")
        if on_dataset_complete:
            on_dataset_complete(name, report)
        return name, report

    tasks = [_run_one(name) for name in dataset_names]
    results = await asyncio.gather(*tasks)

    per_dataset = dict(results)

    return MultiDatasetReport(
        baseline_url=baseline_url,
        target_url=target_url,
        model=model,
        datasets=dataset_names,
        per_dataset=per_dataset,
    )
