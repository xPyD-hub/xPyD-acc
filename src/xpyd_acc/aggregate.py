"""Multi-run aggregation: combine multiple batch reports to find persistent vs flaky divergences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpyd_acc.batch_compare import BatchReport


@dataclass
class AggregatedSample:
    """Aggregated result for a single sample across multiple runs."""

    sample_id: str
    run_count: int
    diverge_count: int
    match_count: int
    consistency_score: float  # diverge_count / run_count (0.0 = always match, 1.0 = always diverge)
    classification: str  # "persistent", "flaky", "stable"
    prompts: list[str] = field(default_factory=list)


@dataclass
class AggregatedReport:
    """Report combining multiple batch comparison runs."""

    total_runs: int
    total_unique_samples: int
    samples: list[AggregatedSample]
    persistent_count: int = 0
    flaky_count: int = 0
    stable_count: int = 0
    persistent_rate: float = 0.0
    flaky_rate: float = 0.0
    stable_rate: float = 0.0

    def to_json(self) -> str:
        """Serialize the aggregated report to JSON."""
        data: dict[str, Any] = {
            "total_runs": self.total_runs,
            "total_unique_samples": self.total_unique_samples,
            "persistent_count": self.persistent_count,
            "flaky_count": self.flaky_count,
            "stable_count": self.stable_count,
            "persistent_rate": self.persistent_rate,
            "flaky_rate": self.flaky_rate,
            "stable_rate": self.stable_rate,
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "run_count": s.run_count,
                    "diverge_count": s.diverge_count,
                    "match_count": s.match_count,
                    "consistency_score": s.consistency_score,
                    "classification": s.classification,
                }
                for s in self.samples
            ],
        }
        return json.dumps(data, indent=2)


def _classify_sample(diverge_count: int, run_count: int) -> str:
    """Classify a sample based on how many runs it diverged in."""
    if diverge_count == 0:
        return "stable"
    if diverge_count == run_count:
        return "persistent"
    return "flaky"


def aggregate_reports(reports: list[BatchReport]) -> AggregatedReport:
    """Aggregate multiple batch reports into a single aggregated report.

    Each report is assumed to be a run of the same dataset. Samples are matched
    by their sample_id.

    Args:
        reports: List of BatchReport objects from multiple runs.

    Returns:
        AggregatedReport with per-sample classification.

    Raises:
        ValueError: If no reports are provided.
    """
    if not reports:
        msg = "At least one report is required"
        raise ValueError(msg)

    run_count = len(reports)

    # Collect per-sample divergence counts
    sample_diverge: dict[str, int] = {}
    sample_appear: dict[str, int] = {}
    sample_prompts: dict[str, list[str]] = {}

    for report in reports:
        for result in report.results:
            sid = result.sample_id
            sample_appear[sid] = sample_appear.get(sid, 0) + 1
            if result.is_divergent():
                sample_diverge[sid] = sample_diverge.get(sid, 0) + 1
            if sid not in sample_prompts:
                sample_prompts[sid] = []
            if result.prompt not in sample_prompts[sid]:
                sample_prompts[sid].append(result.prompt)

    samples: list[AggregatedSample] = []
    for sid in sorted(sample_appear.keys()):
        appear = sample_appear[sid]
        diverge = sample_diverge.get(sid, 0)
        match = appear - diverge
        score = diverge / appear if appear > 0 else 0.0
        classification = _classify_sample(diverge, appear)
        samples.append(AggregatedSample(
            sample_id=sid,
            run_count=appear,
            diverge_count=diverge,
            match_count=match,
            consistency_score=score,
            classification=classification,
            prompts=sample_prompts.get(sid, []),
        ))

    persistent = sum(1 for s in samples if s.classification == "persistent")
    flaky = sum(1 for s in samples if s.classification == "flaky")
    stable = sum(1 for s in samples if s.classification == "stable")
    total = len(samples)

    return AggregatedReport(
        total_runs=run_count,
        total_unique_samples=total,
        samples=samples,
        persistent_count=persistent,
        flaky_count=flaky,
        stable_count=stable,
        persistent_rate=persistent / total if total > 0 else 0.0,
        flaky_rate=flaky / total if total > 0 else 0.0,
        stable_rate=stable / total if total > 0 else 0.0,
    )


def format_aggregated_report(report: AggregatedReport) -> str:
    """Format aggregated report as human-readable text."""
    lines = [
        "=== Multi-Run Aggregation Report ===",
        f"Total runs: {report.total_runs}",
        f"Unique samples: {report.total_unique_samples}",
        "",
        "--- Classification ---",
        f"  Persistent (all diverge): {report.persistent_count} ({report.persistent_rate:.1%})",
        f"  Flaky (some runs diverge):     {report.flaky_count} ({report.flaky_rate:.1%})",
        f"  Stable (all runs match):       {report.stable_count} ({report.stable_rate:.1%})",
    ]

    persistent = [s for s in report.samples if s.classification == "persistent"]
    flaky = [s for s in report.samples if s.classification == "flaky"]

    if persistent:
        lines.append("")
        lines.append("--- Persistent Divergences ---")
        for s in persistent:
            lines.append(f"  Sample {s.sample_id}: {s.diverge_count}/{s.run_count} runs diverged")

    if flaky:
        lines.append("")
        lines.append("--- Flaky Samples ---")
        for s in flaky:
            lines.append(
                f"  Sample {s.sample_id}: {s.diverge_count}/{s.run_count} runs diverged "
                f"(consistency: {s.consistency_score:.2f})"
            )

    return "\n".join(lines)


def load_batch_report_from_json(path: str | Path) -> BatchReport:
    """Load a BatchReport from a JSON file exported by batch-compare --json.

    Args:
        path: Path to the JSON report file.

    Returns:
        BatchReport reconstructed from JSON data.
    """
    from xpyd_acc.batch_compare import SampleResult, compute_report

    data = json.loads(Path(path).read_text())
    results = []
    for r in data.get("results", []):
        results.append(SampleResult(
            sample_id=r["sample_id"],
            prompt=r["prompt"],
            baseline_output=r["baseline_output"],
            target_output=r["target_output"],
            exact_match=r["exact_match"],
            first_divergence_index=r["first_divergence_index"],
            baseline_logprob_at_divergence=r.get("baseline_logprob_at_divergence"),
            target_logprob_at_divergence=r.get("target_logprob_at_divergence"),
            logprob_gap=r.get("logprob_gap"),
            classification=r.get("classification", "unknown"),
            context_length=r.get("context_length", 0),
        ))

    return compute_report(results)
