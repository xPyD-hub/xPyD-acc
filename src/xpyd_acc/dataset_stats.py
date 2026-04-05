"""Dataset statistics analysis for pre-flight inspection."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from xpyd_acc.batch_compare import DatasetSample
from xpyd_acc.templates import PromptTemplate


@dataclass
class LengthStats:
    """Distribution statistics for a sequence of lengths."""

    min: int = 0
    max: int = 0
    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetStatsReport:
    """Full dataset statistics report."""

    sample_count: int = 0
    char_stats: LengthStats = field(default_factory=LengthStats)
    token_stats: LengthStats = field(default_factory=LengthStats)
    duplicate_count: int = 0
    unique_prompts: int = 0
    duplicates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "char_stats": self.char_stats.to_dict(),
            "token_stats": self.token_stats.to_dict(),
            "duplicate_count": self.duplicate_count,
            "unique_prompts": self.unique_prompts,
            "duplicates": self.duplicates,
        }

    def to_json(self, path: str | Path) -> None:
        """Export report as JSON."""
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def estimate_tokens(text: str) -> int:
    """Estimate token count using simple word/0.75 heuristic."""
    words = len(text.split())
    return max(1, math.ceil(words / 0.75)) if words > 0 else 0


def _compute_length_stats(lengths: list[int]) -> LengthStats:
    """Compute distribution stats from a list of lengths."""
    if not lengths:
        return LengthStats()
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    p95_idx = min(math.ceil(0.95 * n) - 1, n - 1)
    return LengthStats(
        min=sorted_lengths[0],
        max=sorted_lengths[-1],
        mean=round(statistics.mean(sorted_lengths), 2),
        median=round(statistics.median(sorted_lengths), 2),
        p95=round(float(sorted_lengths[p95_idx]), 2),
    )


def compute_stats(
    samples: list[DatasetSample],
    template: PromptTemplate | None = None,
) -> DatasetStatsReport:
    """Compute dataset statistics from loaded samples."""
    if not samples:
        return DatasetStatsReport()

    prompts: list[str] = []
    for s in samples:
        if template is not None:
            try:
                prompts.append(template.render(s.metadata if s.metadata else {}))
            except KeyError:
                prompts.append(s.prompt)
        else:
            prompts.append(s.prompt)

    char_lengths = [len(p) for p in prompts]
    token_lengths = [estimate_tokens(p) for p in prompts]

    counter = Counter(prompts)
    duplicates = [
        {"prompt": p[:120], "count": c} for p, c in counter.most_common() if c > 1
    ]

    return DatasetStatsReport(
        sample_count=len(samples),
        char_stats=_compute_length_stats(char_lengths),
        token_stats=_compute_length_stats(token_lengths),
        duplicate_count=sum(c - 1 for c in counter.values() if c > 1),
        unique_prompts=len(counter),
        duplicates=duplicates,
    )


def print_stats(report: DatasetStatsReport, console: Console | None = None) -> None:
    """Print dataset stats to terminal using Rich."""
    console = console or Console()

    console.print(f"\n[bold]Dataset Statistics[/bold]  ({report.sample_count} samples)")
    console.print(f"  Unique prompts: {report.unique_prompts}")
    console.print(f"  Duplicate prompts: {report.duplicate_count}")

    table = Table(title="Prompt Length Distribution")
    table.add_column("Metric", style="cyan")
    table.add_column("Characters", justify="right")
    table.add_column("Est. Tokens", justify="right")

    for label in ("min", "max", "mean", "median", "p95"):
        table.add_row(
            label,
            str(getattr(report.char_stats, label)),
            str(getattr(report.token_stats, label)),
        )
    console.print(table)

    if report.duplicates:
        dup_table = Table(title="Duplicate Prompts")
        dup_table.add_column("Count", justify="right", style="red")
        dup_table.add_column("Prompt (truncated)")
        for d in report.duplicates[:10]:
            dup_table.add_row(str(d["count"]), d["prompt"])
        console.print(dup_table)
