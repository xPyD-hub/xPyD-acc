"""Divergence heatmap by token position — bin divergence points across samples."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HeatmapBucket:
    """One bucket in the position heatmap."""

    range_start: int
    range_end: int  # exclusive
    divergence_count: int
    total_in_range: int  # divergent samples whose divergence_index falls here
    avg_logprob_gap: float | None

    @property
    def label(self) -> str:
        return f"{self.range_start}-{self.range_end - 1}"

    def to_dict(self) -> dict:
        return {
            "range_start": self.range_start,
            "range_end": self.range_end,
            "divergence_count": self.divergence_count,
            "total_in_range": self.total_in_range,
            "avg_logprob_gap": (
                round(self.avg_logprob_gap, 6) if self.avg_logprob_gap is not None else None
            ),
        }


@dataclass
class HeatmapReport:
    """Full heatmap analysis result."""

    buckets: list[HeatmapBucket]
    total_divergent: int
    max_divergence_index: int | None
    num_buckets: int

    def to_dict(self) -> dict:
        return {
            "total_divergent": self.total_divergent,
            "max_divergence_index": self.max_divergence_index,
            "num_buckets": self.num_buckets,
            "buckets": [b.to_dict() for b in self.buckets],
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def compute_heatmap(
    results: list,
    num_buckets: int = 10,
) -> HeatmapReport:
    """Compute divergence heatmap from SampleResult list.

    Args:
        results: list of SampleResult (from BatchReport.results)
        num_buckets: number of position buckets
    """
    if num_buckets < 1:
        num_buckets = 1

    # Collect divergent samples with valid index
    divergent = [
        r for r in results
        if not r.match and r.first_divergence_index is not None
    ]

    if not divergent:
        return HeatmapReport(
            buckets=[],
            total_divergent=0,
            max_divergence_index=None,
            num_buckets=num_buckets,
        )

    max_idx = max(r.first_divergence_index for r in divergent)
    # Bucket width: ceil division so all indices fit
    bucket_width = max(1, math.ceil((max_idx + 1) / num_buckets))

    # Build buckets
    buckets: list[HeatmapBucket] = []
    for i in range(num_buckets):
        start = i * bucket_width
        end = start + bucket_width
        if start > max_idx:
            break
        in_bucket = [
            r for r in divergent
            if start <= r.first_divergence_index < end
        ]
        gaps = [
            r.logprob_gap for r in in_bucket
            if r.logprob_gap is not None
        ]
        avg_gap = sum(gaps) / len(gaps) if gaps else None
        buckets.append(HeatmapBucket(
            range_start=start,
            range_end=end,
            divergence_count=len(in_bucket),
            total_in_range=len(in_bucket),
            avg_logprob_gap=avg_gap,
        ))

    return HeatmapReport(
        buckets=buckets,
        total_divergent=len(divergent),
        max_divergence_index=max_idx,
        num_buckets=num_buckets,
    )


def format_heatmap(report: HeatmapReport) -> str:
    """Format heatmap as a rich terminal string with bar chart."""
    if not report.buckets:
        return "No divergent samples to analyze."

    lines: list[str] = []
    lines.append(f"Divergence Heatmap ({report.total_divergent} divergent samples)")
    lines.append(f"Max divergence index: {report.max_divergence_index}")
    lines.append("")

    max_count = max(b.divergence_count for b in report.buckets)
    bar_width = 40

    # Header
    lines.append(f"{'Position':<14} {'Count':>6}  {'Bar':<{bar_width}}  {'Avg Gap':>10}")
    lines.append("-" * (14 + 6 + 2 + bar_width + 2 + 10))

    for b in report.buckets:
        bar_len = int(b.divergence_count / max_count * bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        gap_str = f"{b.avg_logprob_gap:.4f}" if b.avg_logprob_gap is not None else "N/A"
        lines.append(f"{b.label:<14} {b.divergence_count:>6}  {bar:<{bar_width}}  {gap_str:>10}")

    return "\n".join(lines)
