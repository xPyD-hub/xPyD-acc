"""Regression detection: compare batch results across runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SampleDelta:
    """Change in a single sample between two runs."""

    sample_id: str
    status: str  # "regression", "fix", "persistent_divergence", "persistent_match"
    baseline_match: bool
    current_match: bool
    prompt_preview: str  # truncated prompt for context

    def is_regression(self) -> bool:
        """Sample matched before but diverges now."""
        return self.status == "regression"

    def is_fix(self) -> bool:
        """Sample diverged before but matches now."""
        return self.status == "fix"


@dataclass
class RegressionReport:
    """Report comparing two batch runs."""

    total_samples: int
    regressions: int
    fixes: int
    persistent_divergences: int
    persistent_matches: int
    net_change: int  # positive = improvement (more fixes than regressions)
    baseline_divergence_rate: float
    current_divergence_rate: float
    deltas: list[SampleDelta] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        """Whether any regressions were found."""
        return self.regressions > 0

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = asdict(self)
        data["has_regressions"] = self.has_regressions
        return json.dumps(data, indent=2)


def _load_batch_results(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load batch result JSON and return {sample_id: result_dict}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Batch result file not found: {path}")

    data = json.loads(path.read_text())

    # Support both raw list and BatchReport format
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError(f"Unexpected JSON format in {path}")

    by_id: dict[str, dict[str, Any]] = {}
    for r in results:
        sid = r.get("sample_id", r.get("id", ""))
        if sid:
            by_id[sid] = r
    return by_id


def _truncate(text: str, max_length: int = 80) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def compare_runs(
    baseline_path: str | Path,
    current_path: str | Path,
) -> RegressionReport:
    """Compare two batch result JSONs and produce a regression report."""
    baseline = _load_batch_results(baseline_path)
    current = _load_batch_results(current_path)

    # Use intersection of sample IDs
    common_ids = sorted(set(baseline.keys()) & set(current.keys()))
    if not common_ids:
        return RegressionReport(
            total_samples=0,
            regressions=0,
            fixes=0,
            persistent_divergences=0,
            persistent_matches=0,
            net_change=0,
            baseline_divergence_rate=0.0,
            current_divergence_rate=0.0,
        )

    deltas: list[SampleDelta] = []
    regressions = 0
    fixes = 0
    persistent_divergences = 0
    persistent_matches = 0
    baseline_divergent = 0
    current_divergent = 0

    for sid in common_ids:
        b = baseline[sid]
        c = current[sid]

        b_match = b.get("exact_match", False)
        c_match = c.get("exact_match", False)

        if not b_match:
            baseline_divergent += 1
        if not c_match:
            current_divergent += 1

        prompt = b.get("prompt", c.get("prompt", ""))

        if b_match and c_match:
            status = "persistent_match"
            persistent_matches += 1
        elif b_match and not c_match:
            status = "regression"
            regressions += 1
        elif not b_match and c_match:
            status = "fix"
            fixes += 1
        else:
            status = "persistent_divergence"
            persistent_divergences += 1

        deltas.append(
            SampleDelta(
                sample_id=sid,
                status=status,
                baseline_match=b_match,
                current_match=c_match,
                prompt_preview=_truncate(prompt),
            )
        )

    total = len(common_ids)
    return RegressionReport(
        total_samples=total,
        regressions=regressions,
        fixes=fixes,
        persistent_divergences=persistent_divergences,
        persistent_matches=persistent_matches,
        net_change=fixes - regressions,
        baseline_divergence_rate=baseline_divergent / total if total else 0.0,
        current_divergence_rate=current_divergent / total if total else 0.0,
        deltas=deltas,
    )


def format_regression_report(report: RegressionReport) -> str:
    """Format regression report for terminal output."""
    lines: list[str] = []
    status = "❌ REGRESSIONS FOUND" if report.has_regressions else "✅ NO REGRESSIONS"
    lines.append(f"Regression Report: {status}")
    lines.append("")
    lines.append(f"  Total samples compared:  {report.total_samples}")
    lines.append(f"  Regressions:             {report.regressions}")
    lines.append(f"  Fixes:                   {report.fixes}")
    lines.append(f"  Persistent divergences:  {report.persistent_divergences}")
    lines.append(f"  Persistent matches:      {report.persistent_matches}")
    lines.append(f"  Net change:              {report.net_change:+d}")
    lines.append("")
    lines.append(
        f"  Baseline divergence rate: {report.baseline_divergence_rate:.1%}"
    )
    lines.append(
        f"  Current divergence rate:  {report.current_divergence_rate:.1%}"
    )

    # Show regressions
    regression_deltas = [d for d in report.deltas if d.is_regression()]
    if regression_deltas:
        lines.append("")
        lines.append("  Regressions:")
        for d in regression_deltas[:20]:
            lines.append(f"    ❌ {d.sample_id}: {d.prompt_preview}")
        if len(regression_deltas) > 20:
            lines.append(f"    ... and {len(regression_deltas) - 20} more")

    # Show fixes
    fix_deltas = [d for d in report.deltas if d.is_fix()]
    if fix_deltas:
        lines.append("")
        lines.append("  Fixes:")
        for d in fix_deltas[:10]:
            lines.append(f"    ✅ {d.sample_id}: {d.prompt_preview}")
        if len(fix_deltas) > 10:
            lines.append(f"    ... and {len(fix_deltas) - 10} more")

    return "\n".join(lines)
