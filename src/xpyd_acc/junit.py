"""JUnit XML export for batch comparison reports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xpyd_acc.batch_compare import BatchReport, MultiTargetBatchReport, SampleResult


def _sample_to_testcase(result: "SampleResult") -> ET.Element:
    """Convert a SampleResult to a JUnit <testcase> element."""
    tc = ET.Element("testcase")
    tc.set("name", result.sample_id)
    tc.set("classname", "xpyd-acc.batch-compare")

    if result.is_divergent():
        failure = ET.SubElement(tc, "failure")
        parts: list[str] = []
        if result.first_divergence_index is not None:
            parts.append(f"Divergence at token index {result.first_divergence_index}")
        if result.logprob_gap is not None:
            parts.append(f"Logprob gap: {result.logprob_gap:.4f}")
        # Truncated output diff
        baseline_preview = (result.baseline_output or "")[:200]
        target_preview = (result.target_output or "")[:200]
        parts.append(f"Baseline: {baseline_preview}")
        parts.append(f"Target: {target_preview}")
        failure.set("message", parts[0] if parts else "Divergent output")
        failure.text = "\n".join(parts)

    return tc


def report_to_junit(
    report: "BatchReport",
    *,
    suite_name: str = "xpyd-acc",
    timestamp: str | None = None,
) -> str:
    """Convert a BatchReport to JUnit XML string.

    Args:
        report: The batch report to convert.
        suite_name: Name for the <testsuite> element.
        timestamp: ISO timestamp; defaults to current UTC time.

    Returns:
        JUnit XML as a string.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    suite = ET.Element("testsuite")
    suite.set("name", suite_name)
    suite.set("tests", str(report.total_samples))
    suite.set("failures", str(report.divergent_samples))
    suite.set("errors", "0")
    suite.set("timestamp", timestamp)

    for r in report.results:
        suite.append(_sample_to_testcase(r))

    tree = ET.ElementTree(suite)
    ET.indent(tree, space="  ")
    return ET.tostring(suite, encoding="unicode", xml_declaration=True)


def multi_report_to_junit(
    multi_report: "MultiTargetBatchReport",
    *,
    timestamp: str | None = None,
) -> str:
    """Convert a MultiTargetBatchReport to JUnit XML with one testsuite per target.

    Returns:
        JUnit XML as a string.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    root = ET.Element("testsuites")

    for url in multi_report.target_urls:
        report = multi_report.per_target[url]
        suite = ET.Element("testsuite")
        suite.set("name", f"xpyd-acc:{url}")
        suite.set("tests", str(report.total_samples))
        suite.set("failures", str(report.divergent_samples))
        suite.set("errors", "0")
        suite.set("timestamp", timestamp)

        for r in report.results:
            suite.append(_sample_to_testcase(r))

        root.append(suite)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)
