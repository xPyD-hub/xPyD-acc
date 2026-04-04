"""Selective sample rerun: rerun only divergent samples from a previous batch report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xpyd_acc.batch_compare import BatchReport, DatasetSample, SampleResult, compute_report


@dataclass
class RerunPlan:
    """Plan for which samples to rerun."""

    divergent_samples: list[DatasetSample]
    total_in_report: int
    divergent_count: int


def load_divergent_samples(report_path: str | Path) -> RerunPlan:
    """Load a previous JSON report and extract divergent samples for rerun.

    Args:
        report_path: Path to a batch comparison JSON report.

    Returns:
        RerunPlan with the divergent samples as DatasetSample objects.

    Raises:
        FileNotFoundError: If the report file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the report is missing required fields or has no divergent samples.
    """
    path = Path(report_path)
    if not path.exists():
        msg = f"Report file not found: {path}"
        raise FileNotFoundError(msg)

    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in report file: {path}"
        raise json.JSONDecodeError(msg, exc.doc, exc.pos) from exc

    if not isinstance(data, dict) or "results" not in data:
        msg = "Report JSON must contain a 'results' array"
        raise ValueError(msg)

    results = data["results"]
    if not isinstance(results, list):
        msg = "Report 'results' must be a list"
        raise ValueError(msg)

    divergent: list[DatasetSample] = []
    for r in results:
        if not r.get("exact_match", True):
            divergent.append(
                DatasetSample(
                    id=r.get("sample_id", "unknown"),
                    prompt=r.get("prompt", ""),
                    metadata=r.get("metadata", {}),
                )
            )

    if not divergent:
        msg = "No divergent samples found in report — nothing to rerun"
        raise ValueError(msg)

    return RerunPlan(
        divergent_samples=divergent,
        total_in_report=data.get("total_samples", len(results)),
        divergent_count=len(divergent),
    )


def merge_rerun_results(
    original_path: str | Path,
    rerun_report: BatchReport,
) -> BatchReport:
    """Merge rerun results back into the original report.

    Replaces original results for rerun sample IDs with the new results,
    then recomputes report statistics.

    Args:
        original_path: Path to the original JSON report.
        rerun_report: The report from the rerun.

    Returns:
        A new BatchReport combining original + rerun results.
    """
    path = Path(original_path)
    data = json.loads(path.read_text())

    # Build lookup of rerun results by sample_id
    rerun_by_id: dict[str, SampleResult] = {
        r.sample_id: r for r in rerun_report.results
    }

    # Rebuild results: replace rerun samples, keep others
    original_results = _parse_sample_results(data["results"])
    merged: list[SampleResult] = []
    for r in original_results:
        if r.sample_id in rerun_by_id:
            merged.append(rerun_by_id[r.sample_id])
        else:
            merged.append(r)

    return compute_report(merged)


def _parse_sample_results(results_data: list[dict[str, Any]]) -> list[SampleResult]:
    """Parse raw JSON result dicts into SampleResult objects."""
    parsed: list[SampleResult] = []
    for r in results_data:
        parsed.append(
            SampleResult(
                sample_id=r["sample_id"],
                prompt=r["prompt"],
                baseline_output=r["baseline_output"],
                target_output=r["target_output"],
                exact_match=r["exact_match"],
                first_divergence_index=r.get("first_divergence_index"),
                baseline_logprob_at_divergence=r.get("baseline_logprob_at_divergence"),
                target_logprob_at_divergence=r.get("target_logprob_at_divergence"),
                logprob_gap=r.get("logprob_gap"),
                classification=r.get("classification", "unknown"),
                context_length=r.get("context_length", 0),
            )
        )
    return parsed
