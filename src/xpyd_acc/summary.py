"""Compact summary output for batch reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SummaryData:
    """Compact summary extracted from a batch report."""

    dataset: str
    total_samples: int
    divergent_samples: int
    divergence_rate: float
    divergence_index_mean: float | None
    divergence_index_median: float | None
    targets: list[str] | None  # None for single-target

    def to_oneline(self) -> str:
        """Format as a single-line summary string."""
        parts = [
            self.dataset,
            f"{self.total_samples} samples",
            f"{self.divergent_samples} divergent ({self.divergence_rate:.1%})",
        ]
        if self.divergence_index_mean is not None:
            parts.append(f"mean div index: {self.divergence_index_mean:.1f}")
        return " | ".join(parts)

    def to_json(self) -> str:
        """Format as compact single-line JSON."""
        d: dict[str, Any] = {
            "dataset": self.dataset,
            "total_samples": self.total_samples,
            "divergent_samples": self.divergent_samples,
            "divergence_rate": round(self.divergence_rate, 4),
        }
        if self.divergence_index_mean is not None:
            d["divergence_index_mean"] = round(self.divergence_index_mean, 1)
        if self.divergence_index_median is not None:
            d["divergence_index_median"] = round(self.divergence_index_median, 1)
        if self.targets is not None:
            d["targets"] = self.targets
        return json.dumps(d, separators=(",", ":"))

    def to_kv(self) -> str:
        """Format as key=value pairs, one per line."""
        lines = [
            f"dataset={self.dataset}",
            f"total_samples={self.total_samples}",
            f"divergent_samples={self.divergent_samples}",
            f"divergence_rate={self.divergence_rate:.4f}",
        ]
        if self.divergence_index_mean is not None:
            lines.append(f"divergence_index_mean={self.divergence_index_mean:.1f}")
        if self.divergence_index_median is not None:
            lines.append(f"divergence_index_median={self.divergence_index_median:.1f}")
        if self.targets is not None:
            lines.append(f"targets={','.join(self.targets)}")
        return "\n".join(lines)

    def format(self, fmt: str = "oneline") -> str:
        """Format the summary in the requested format."""
        formatters = {
            "oneline": self.to_oneline,
            "json": self.to_json,
            "kv": self.to_kv,
        }
        formatter = formatters.get(fmt)
        if formatter is None:
            msg = f"Unknown format: {fmt!r} (choose from: {', '.join(formatters)})"
            raise ValueError(msg)
        return formatter()


def extract_summary(report_data: dict[str, Any]) -> SummaryData:
    """Extract a SummaryData from a parsed batch report dict."""
    dataset = report_data.get("dataset", "unknown")
    total = report_data.get("total_samples", 0)
    divergent = report_data.get("divergent_samples", 0)
    rate = report_data.get("divergence_rate", 0.0)
    mean_idx = report_data.get("divergence_index_mean")
    median_idx = report_data.get("divergence_index_median")

    # Multi-target reports have per_target key
    targets = None
    if "per_target" in report_data and isinstance(report_data["per_target"], dict):
        targets = list(report_data["per_target"].keys())

    return SummaryData(
        dataset=dataset,
        total_samples=total,
        divergent_samples=divergent,
        divergence_rate=rate,
        divergence_index_mean=mean_idx,
        divergence_index_median=median_idx,
        targets=targets,
    )


def load_and_summarize(path: str | Path, fmt: str = "oneline") -> str:
    """Load a JSON report file and return a formatted summary string."""
    report_path = Path(path)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = extract_summary(data)
    return summary.format(fmt)
