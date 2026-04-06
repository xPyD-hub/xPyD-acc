"""Generate reusable test datasets from divergent samples in batch reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .batch_compare import BatchReport
from .log import get_logger

logger = get_logger(__name__)


@dataclass
class SuiteEntry:
    """A single entry in a generated test suite."""

    id: str
    prompt: str
    expected: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None expected."""
        d: dict[str, Any] = {"id": self.id, "prompt": self.prompt}
        if self.expected is not None:
            d["expected"] = self.expected
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSONL line."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class GenerateSuiteConfig:
    """Configuration for test suite generation filters."""

    classification: str | None = None
    deterministic_only: bool = False
    min_logprob_gap: float | None = None
    max_samples: int | None = None
    include_expected: bool = False


def generate_suite(
    report: BatchReport,
    config: GenerateSuiteConfig | None = None,
) -> list[SuiteEntry]:
    """Generate a test suite from divergent samples in a batch report.

    Args:
        report: A BatchReport containing comparison results.
        config: Optional filtering configuration.

    Returns:
        List of SuiteEntry objects for the generated test suite.
    """
    if config is None:
        config = GenerateSuiteConfig()

    entries: list[SuiteEntry] = []

    for result in report.results:
        # Only include divergent samples
        if not result.is_divergent():
            continue

        # Filter by classification
        if config.classification and result.classification != config.classification:
            continue

        # Filter by minimum logprob gap
        if config.min_logprob_gap is not None:
            if result.logprob_gap is None or result.logprob_gap < config.min_logprob_gap:
                continue

        metadata: dict[str, Any] = {
            "classification": result.classification,
            "divergence_index": result.first_divergence_index,
            "logprob_gap": result.logprob_gap,
            "context_length": result.context_length,
        }

        expected = result.baseline_output if config.include_expected else None

        entries.append(
            SuiteEntry(
                id=result.sample_id,
                prompt=result.prompt,
                expected=expected,
                metadata=metadata,
            )
        )

    # Cap the number of samples
    if config.max_samples is not None and len(entries) > config.max_samples:
        entries = entries[: config.max_samples]

    logger.info("Generated test suite with %d entries from %d divergent samples",
                len(entries), report.divergent_samples)
    return entries


def write_suite(entries: list[SuiteEntry], output: Path) -> None:
    """Write test suite entries to a JSONL file.

    Args:
        entries: List of SuiteEntry objects.
        output: Path to the output JSONL file.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_jsonl_line() + "\n")
    logger.info("Wrote %d entries to %s", len(entries), output)


def format_suite_summary(entries: list[SuiteEntry], report: BatchReport) -> str:
    """Format a human-readable summary of the generated test suite.

    Args:
        entries: Generated suite entries.
        report: The source batch report.

    Returns:
        Formatted summary string.
    """
    lines = [
        "Test Suite Generation Summary",
        "=" * 40,
        f"Source report: {report.total_samples} total samples, "
        f"{report.divergent_samples} divergent",
        f"Generated suite: {len(entries)} samples",
        "",
    ]

    if entries:
        classifications: dict[str, int] = {}
        for entry in entries:
            cls = entry.metadata.get("classification", "unknown")
            classifications[cls] = classifications.get(cls, 0) + 1

        lines.append("By classification:")
        for cls, count in sorted(classifications.items()):
            lines.append(f"  {cls}: {count}")

        has_expected = sum(1 for e in entries if e.expected is not None)
        lines.append(f"\nWith expected output: {has_expected}/{len(entries)}")
    else:
        lines.append("No samples matched the filter criteria.")

    return "\n".join(lines)
