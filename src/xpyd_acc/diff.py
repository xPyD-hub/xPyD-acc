"""Report diff: side-by-side comparison of two batch reports with output text changes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import Any


@dataclass
class SampleTransition:
    """Transition of a single sample between two reports."""

    sample_id: str
    status: str  # "regression", "fix", "unchanged_match", "unchanged_diverge", "new", "removed"
    old_match: bool | None  # None if new
    new_match: bool | None  # None if removed
    old_output: str | None
    new_output: str | None
    output_diff: str | None  # unified diff of outputs, None if identical or unavailable
    prompt_preview: str


@dataclass
class DiffResult:
    """Result of diffing two batch reports."""

    total_old: int
    total_new: int
    common: int
    regressions: int
    fixes: int
    unchanged_match: int
    unchanged_diverge: int
    new_samples: int
    removed_samples: int
    output_changes: int  # samples where output text changed (regardless of match status)
    transitions: list[SampleTransition] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)


def _load_report(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load batch report and return {sample_id: result_dict}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Report file not found: {path}")

    data = json.loads(path.read_text())

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
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _get_output(sample: dict[str, Any]) -> str:
    """Extract target output text from a sample result."""
    return sample.get("target_output", sample.get("output", ""))


def _make_diff(old_text: str, new_text: str) -> str | None:
    """Generate unified diff between two texts. Returns None if identical."""
    if old_text == new_text:
        return None
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff_lines = list(unified_diff(old_lines, new_lines, fromfile="old", tofile="new"))
    if not diff_lines:
        return None
    return "".join(diff_lines)


def diff_reports(
    old_path: str | Path,
    new_path: str | Path,
) -> DiffResult:
    """Compare two batch reports and produce a detailed diff."""
    old_data = _load_report(old_path)
    new_data = _load_report(new_path)

    old_ids = set(old_data.keys())
    new_ids = set(new_data.keys())
    common_ids = sorted(old_ids & new_ids)
    new_only = sorted(new_ids - old_ids)
    removed_only = sorted(old_ids - new_ids)

    transitions: list[SampleTransition] = []
    regressions = fixes = unchanged_match = unchanged_diverge = output_changes = 0

    for sid in common_ids:
        o = old_data[sid]
        n = new_data[sid]
        o_match = o.get("exact_match", False)
        n_match = n.get("exact_match", False)
        o_output = _get_output(o)
        n_output = _get_output(n)
        prompt = o.get("prompt", n.get("prompt", ""))
        text_diff = _make_diff(o_output, n_output)

        if text_diff is not None:
            output_changes += 1

        if o_match and n_match:
            status = "unchanged_match"
            unchanged_match += 1
        elif o_match and not n_match:
            status = "regression"
            regressions += 1
        elif not o_match and n_match:
            status = "fix"
            fixes += 1
        else:
            status = "unchanged_diverge"
            unchanged_diverge += 1

        transitions.append(SampleTransition(
            sample_id=sid,
            status=status,
            old_match=o_match,
            new_match=n_match,
            old_output=o_output,
            new_output=n_output,
            output_diff=text_diff,
            prompt_preview=_truncate(prompt),
        ))

    for sid in new_only:
        n = new_data[sid]
        prompt = n.get("prompt", "")
        transitions.append(SampleTransition(
            sample_id=sid,
            status="new",
            old_match=None,
            new_match=n.get("exact_match", False),
            old_output=None,
            new_output=_get_output(n),
            output_diff=None,
            prompt_preview=_truncate(prompt),
        ))

    for sid in removed_only:
        o = old_data[sid]
        prompt = o.get("prompt", "")
        transitions.append(SampleTransition(
            sample_id=sid,
            status="removed",
            old_match=o.get("exact_match", False),
            new_match=None,
            old_output=_get_output(o),
            new_output=None,
            output_diff=None,
            prompt_preview=_truncate(prompt),
        ))

    return DiffResult(
        total_old=len(old_data),
        total_new=len(new_data),
        common=len(common_ids),
        regressions=regressions,
        fixes=fixes,
        unchanged_match=unchanged_match,
        unchanged_diverge=unchanged_diverge,
        new_samples=len(new_only),
        removed_samples=len(removed_only),
        output_changes=output_changes,
        transitions=transitions,
    )


def format_diff_report(result: DiffResult) -> str:
    """Format diff result for terminal output."""
    lines: list[str] = []
    lines.append("Report Diff Summary")
    lines.append("=" * 40)
    lines.append(f"  Old report samples:   {result.total_old}")
    lines.append(f"  New report samples:   {result.total_new}")
    lines.append(f"  Common samples:       {result.common}")
    lines.append(f"  New samples:          {result.new_samples}")
    lines.append(f"  Removed samples:      {result.removed_samples}")
    lines.append("")
    lines.append("Transitions:")
    lines.append(f"  ❌ Regressions:       {result.regressions}")
    lines.append(f"  ✅ Fixes:             {result.fixes}")
    lines.append(f"  ➡️  Unchanged match:   {result.unchanged_match}")
    lines.append(f"  ➡️  Unchanged diverge: {result.unchanged_diverge}")
    lines.append(f"  📝 Output changes:    {result.output_changes}")

    # Show regressions with diffs
    reg = [t for t in result.transitions if t.status == "regression"]
    if reg:
        lines.append("")
        lines.append("Regressions:")
        for t in reg[:10]:
            lines.append(f"  ❌ {t.sample_id}: {t.prompt_preview}")
            if t.output_diff:
                for dl in t.output_diff.splitlines()[:6]:
                    lines.append(f"      {dl}")
        if len(reg) > 10:
            lines.append(f"  ... and {len(reg) - 10} more")

    # Show fixes
    fix = [t for t in result.transitions if t.status == "fix"]
    if fix:
        lines.append("")
        lines.append("Fixes:")
        for t in fix[:10]:
            lines.append(f"  ✅ {t.sample_id}: {t.prompt_preview}")
        if len(fix) > 10:
            lines.append(f"  ... and {len(fix) - 10} more")

    return "\n".join(lines)
