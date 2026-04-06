"""Offline file-based comparison: compare pre-collected outputs without endpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from xpyd_acc.batch_compare import BatchReport, SampleResult, compute_report
from xpyd_acc.log import get_logger
from xpyd_acc.output_compare import MatchConfig, normalized_match

logger = get_logger("file_compare")


@dataclass
class FileOutput:
    """A single output loaded from a JSONL file."""

    id: str
    output: str
    logprobs: list[float] | None = None


def load_outputs(path: Path) -> list[FileOutput]:
    """Load outputs from a JSONL file.

    Each line must be a JSON object with at least ``id`` and ``output`` fields.
    An optional ``logprobs`` field (list of floats) is supported.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is missing required fields or is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    outputs: list[FileOutput] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_num}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                typ = type(obj).__name__
                raise ValueError(
                    f"{path}:{line_num}: expected JSON object, got {typ}"
                )
            if "id" not in obj:
                raise ValueError(f"{path}:{line_num}: missing required field 'id'")
            if "output" not in obj:
                raise ValueError(f"{path}:{line_num}: missing required field 'output'")

            outputs.append(FileOutput(
                id=str(obj["id"]),
                output=str(obj["output"]),
                logprobs=obj.get("logprobs"),
            ))

    if not outputs:
        raise ValueError(f"No samples found in {path}")

    return outputs


def _estimate_context_length(text: str) -> int:
    """Rough token count estimate (words / 0.75)."""
    words = len(text.split())
    return max(1, int(words / 0.75))


def run_file_compare(
    baseline_outputs: list[FileOutput],
    target_outputs: list[FileOutput],
    *,
    match_config: MatchConfig | None = None,
    logprob_gap_threshold: float = 0.1,
) -> BatchReport:
    """Compare baseline and target outputs loaded from files.

    Matches samples by ID. Both lists must contain the same set of IDs.

    Returns:
        A :class:`BatchReport` with comparison results.

    Raises:
        ValueError: If IDs don't match between baseline and target.
    """
    if match_config is None:
        match_config = MatchConfig()

    baseline_map = {o.id: o for o in baseline_outputs}
    target_map = {o.id: o for o in target_outputs}

    baseline_ids = set(baseline_map.keys())
    target_ids = set(target_map.keys())

    if baseline_ids != target_ids:
        only_baseline = baseline_ids - target_ids
        only_target = target_ids - baseline_ids
        parts = []
        if only_baseline:
            parts.append(f"only in baseline: {sorted(only_baseline)[:5]}")
        if only_target:
            parts.append(f"only in target: {sorted(only_target)[:5]}")
        raise ValueError(f"Sample ID mismatch: {'; '.join(parts)}")

    results: list[SampleResult] = []

    for sample_id in sorted(baseline_ids):
        bl = baseline_map[sample_id]
        tg = target_map[sample_id]

        exact = normalized_match(bl.output, tg.output, match_config)

        # Find first divergence index (character-level)
        first_div_idx: int | None = None
        if not exact:
            bl_tokens = bl.output.split()
            tg_tokens = tg.output.split()
            for i, (bt, tt) in enumerate(zip(bl_tokens, tg_tokens)):
                if bt != tt:
                    first_div_idx = i
                    break
            else:
                # One is a prefix of the other
                first_div_idx = min(len(bl_tokens), len(tg_tokens))

        # Logprob gap at divergence point
        bl_logprob: float | None = None
        tg_logprob: float | None = None
        logprob_gap: float | None = None
        if first_div_idx is not None and bl.logprobs and tg.logprobs:
            if first_div_idx < len(bl.logprobs):
                bl_logprob = bl.logprobs[first_div_idx]
            if first_div_idx < len(tg.logprobs):
                tg_logprob = tg.logprobs[first_div_idx]
            if bl_logprob is not None and tg_logprob is not None:
                logprob_gap = abs(bl_logprob - tg_logprob)

        # Classification
        if exact:
            classification = "match"
        elif logprob_gap is not None:
            if logprob_gap >= logprob_gap_threshold:
                classification = "likely_bug"
            else:
                classification = "likely_uncertainty"
        else:
            classification = "unknown"

        results.append(SampleResult(
            sample_id=sample_id,
            prompt=f"[file:{sample_id}]",
            baseline_output=bl.output,
            target_output=tg.output,
            exact_match=exact,
            first_divergence_index=first_div_idx,
            baseline_logprob_at_divergence=bl_logprob,
            target_logprob_at_divergence=tg_logprob,
            logprob_gap=logprob_gap,
            classification=classification,
            context_length=_estimate_context_length(bl.output),
        ))

    return compute_report(results, logprob_gap_threshold=logprob_gap_threshold)


def format_file_compare(report: BatchReport) -> str:
    """Format a file comparison report for terminal display."""
    lines = [
        "═══ File Comparison Report ═══",
        "",
        f"Total samples:     {report.total_samples}",
        f"Matching:          {report.match_samples}",
        f"Divergent:         {report.divergent_samples}",
        f"Divergence rate:   {report.divergence_rate:.1%}",
    ]

    if report.likely_bugs:
        lines.append(f"Likely bugs:       {report.likely_bugs}")
    if report.likely_uncertainty:
        lines.append(f"Likely uncertainty: {report.likely_uncertainty}")
    if report.unknown_classification:
        lines.append(f"Unknown:           {report.unknown_classification}")

    if report.divergence_index_mean is not None:
        lines.append(f"Avg divergence idx: {report.divergence_index_mean:.1f}")
    if report.logprob_gap_mean is not None:
        lines.append(f"Avg logprob gap:   {report.logprob_gap_mean:.4f}")

    return "\n".join(lines)
