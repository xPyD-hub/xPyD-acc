"""Multi-model comparison: run the same dataset against multiple models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from xpyd_acc.batch_compare import BatchReport, DatasetSample, run_batch
from xpyd_acc.log import get_logger

logger = get_logger("multi_model")


@dataclass
class CrossModelSummary:
    """Cross-model divergence analysis."""

    # sample_id -> list of models where the sample diverged
    per_sample_divergent_models: dict[str, list[str]] = field(default_factory=dict)
    # Samples that diverge in ALL models (systematic issue)
    systematic_divergent_ids: list[str] = field(default_factory=list)
    # Samples that diverge in some but not all models (model-specific)
    model_specific_divergent_ids: list[str] = field(default_factory=list)
    # Samples that match in all models
    all_match_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "per_sample_divergent_models": self.per_sample_divergent_models,
            "systematic_divergent_count": len(self.systematic_divergent_ids),
            "systematic_divergent_ids": self.systematic_divergent_ids,
            "model_specific_divergent_count": len(self.model_specific_divergent_ids),
            "model_specific_divergent_ids": self.model_specific_divergent_ids,
            "all_match_count": len(self.all_match_ids),
        }


@dataclass
class MultiModelBatchReport:
    """Report comparing a dataset across multiple models."""

    baseline_url: str
    target_url: str
    models: list[str]
    per_model: dict[str, BatchReport]
    cross_model: CrossModelSummary
    total_samples: int

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data: dict[str, Any] = {
            "baseline_url": self.baseline_url,
            "target_url": self.target_url,
            "models": self.models,
            "total_samples": self.total_samples,
            "per_model": {
                model: json.loads(report.to_json())
                for model, report in self.per_model.items()
            },
            "cross_model": self.cross_model.to_dict(),
        }
        return json.dumps(data, indent=2)

    def to_markdown(self, *, max_divergent_samples: int = 10) -> str:
        """Serialize to Markdown string."""
        lines: list[str] = []
        lines.append("# Multi-Model Batch Comparison Report")
        lines.append("")
        lines.append(f"**Baseline:** `{self.baseline_url}`")
        lines.append(f"**Target:** `{self.target_url}`")
        lines.append(f"**Models:** {', '.join(f'`{m}`' for m in self.models)}")
        lines.append(f"**Total samples:** {self.total_samples}")
        lines.append("")

        # Per-model summary table
        lines.append("## Per-Model Summary")
        lines.append("")
        lines.append("| Model | Matches | Divergent | Rate |")
        lines.append("|-------|---------|-----------|------|")
        for model in self.models:
            r = self.per_model[model]
            lines.append(
                f"| `{model}` | {r.match_samples} | {r.divergent_samples} "
                f"| {r.divergence_rate:.1%} |"
            )
        lines.append("")

        # Cross-model summary
        cm = self.cross_model
        lines.append("## Cross-Model Analysis")
        lines.append("")
        lines.append(
            f"- **Systematic divergences** (all models): "
            f"{len(cm.systematic_divergent_ids)}"
        )
        lines.append(
            f"- **Model-specific divergences** (some models): "
            f"{len(cm.model_specific_divergent_ids)}"
        )
        lines.append(f"- **All match** (no model diverges): {len(cm.all_match_ids)}")
        lines.append("")

        # Systematic divergences detail
        if cm.systematic_divergent_ids:
            show = cm.systematic_divergent_ids[:max_divergent_samples]
            lines.append("### Systematic Divergences")
            lines.append("")
            for sid in show:
                lines.append(f"- `{sid}`")
            if len(cm.systematic_divergent_ids) > max_divergent_samples:
                lines.append(
                    f"- ... and {len(cm.systematic_divergent_ids) - max_divergent_samples} more"
                )
            lines.append("")

        return "\n".join(lines)


def compute_cross_model_summary(
    models: list[str],
    per_model: dict[str, BatchReport],
) -> CrossModelSummary:
    """Compute cross-model divergence analysis from per-model reports."""
    # Collect all sample IDs from the first model's results
    if not models or not per_model:
        return CrossModelSummary()

    first_report = per_model[models[0]]
    sample_ids = [r.sample_id for r in first_report.results]

    # Build per-sample divergence map
    per_sample: dict[str, list[str]] = {}
    for sid in sample_ids:
        divergent_in: list[str] = []
        for model in models:
            report = per_model[model]
            for result in report.results:
                if result.sample_id == sid:
                    if result.is_divergent():
                        divergent_in.append(model)
                    break
        per_sample[sid] = divergent_in

    systematic = [
        sid for sid, divs in per_sample.items()
        if len(divs) == len(models) and len(divs) > 0
    ]
    model_specific = [sid for sid, divs in per_sample.items() if 0 < len(divs) < len(models)]
    all_match = [sid for sid, divs in per_sample.items() if len(divs) == 0]

    return CrossModelSummary(
        per_sample_divergent_models=per_sample,
        systematic_divergent_ids=systematic,
        model_specific_divergent_ids=model_specific,
        all_match_ids=all_match,
    )


def format_multi_model_report(report: MultiModelBatchReport) -> str:
    """Format multi-model report for terminal display."""
    lines: list[str] = []
    lines.append("Multi-Model Batch Comparison")
    lines.append(f"  Baseline: {report.baseline_url}")
    lines.append(f"  Target:   {report.target_url}")
    lines.append(f"  Models:   {', '.join(report.models)}")
    lines.append(f"  Samples:  {report.total_samples}")
    lines.append("")

    for model in report.models:
        r = report.per_model[model]
        status = "✅" if r.divergent_samples == 0 else "❌"
        lines.append(
            f"  {status} {model}: {r.match_samples}/{r.total_samples} match "
            f"({r.divergence_rate:.1%} divergence)"
        )

    cm = report.cross_model
    lines.append("")
    lines.append(f"  Systematic (all models):  {len(cm.systematic_divergent_ids)}")
    lines.append(f"  Model-specific:           {len(cm.model_specific_divergent_ids)}")
    lines.append(f"  All match:                {len(cm.all_match_ids)}")

    return "\n".join(lines)


async def run_multi_model(
    samples: list[DatasetSample],
    baseline_url: str,
    target_url: str,
    models: list[str],
    *,
    max_tokens: int = 64,
    api_key: str = "no-key",
    logprob_gap_threshold: float = 0.1,
    concurrency: int = 5,
    retries: int = 3,
    retry_delay: float = 1.0,
    on_model_complete: Callable[[str, BatchReport], None] | None = None,
    match_config: Any | None = None,
    sampling_params: Any | None = None,
    timeout: float = 120.0,
    skip_validation: bool = False,
    custom_headers: dict[str, str] | None = None,
) -> MultiModelBatchReport:
    """Run batch comparison for each model and produce a multi-model report.

    Args:
        models: List of model names to compare.
        on_model_complete: Optional callback after each model completes.
    """
    per_model: dict[str, BatchReport] = {}

    for model in models:
        logger.info("Running batch for model: %s", model)
        report = await run_batch(
            samples,
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
        per_model[model] = report
        if on_model_complete:
            on_model_complete(model, report)

    cross_model = compute_cross_model_summary(models, per_model)

    return MultiModelBatchReport(
        baseline_url=baseline_url,
        target_url=target_url,
        models=models,
        per_model=per_model,
        cross_model=cross_model,
        total_samples=len(samples),
    )
