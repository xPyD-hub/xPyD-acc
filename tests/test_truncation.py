"""Tests for M61: Output Truncation Detection."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_acc.batch_compare import (
    SampleResult,
    compute_report,
    export_markdown,
    format_report,
    load_report,
)


def _make_result(
    sample_id: str = "0",
    exact_match: bool = True,
    baseline_finish_reason: str | None = "stop",
    target_finish_reason: str | None = "stop",
) -> SampleResult:
    """Helper to create a SampleResult with finish_reason fields."""
    return SampleResult(
        sample_id=sample_id,
        prompt="test prompt",
        baseline_output="hello world",
        target_output="hello world" if exact_match else "hello earth",
        exact_match=exact_match,
        first_divergence_index=None if exact_match else 1,
        baseline_logprob_at_divergence=None,
        target_logprob_at_divergence=None,
        logprob_gap=None,
        classification="match" if exact_match else "unknown",
        context_length=2,
        baseline_finish_reason=baseline_finish_reason,
        target_finish_reason=target_finish_reason,
    )


class TestSampleResultFinishReason:
    """Tests for finish_reason fields on SampleResult."""

    def test_default_none(self) -> None:
        r = SampleResult(
            sample_id="0", prompt="p", baseline_output="a", target_output="a",
            exact_match=True, first_divergence_index=None,
            baseline_logprob_at_divergence=None, target_logprob_at_divergence=None,
            logprob_gap=None, classification="match", context_length=1,
        )
        assert r.baseline_finish_reason is None
        assert r.target_finish_reason is None

    def test_set_finish_reason(self) -> None:
        r = _make_result(baseline_finish_reason="length", target_finish_reason="stop")
        assert r.baseline_finish_reason == "length"
        assert r.target_finish_reason == "stop"

    def test_both_length(self) -> None:
        r = _make_result(baseline_finish_reason="length", target_finish_reason="length")
        assert r.baseline_finish_reason == "length"
        assert r.target_finish_reason == "length"


class TestBatchReportTruncatedCount:
    """Tests for truncated_count in BatchReport."""

    def test_no_truncation(self) -> None:
        results = [_make_result(sample_id=str(i)) for i in range(3)]
        report = compute_report(results)
        assert report.truncated_count == 0

    def test_baseline_truncated(self) -> None:
        results = [
            _make_result(sample_id="0", baseline_finish_reason="length"),
            _make_result(sample_id="1"),
        ]
        report = compute_report(results)
        assert report.truncated_count == 1

    def test_target_truncated(self) -> None:
        results = [
            _make_result(sample_id="0", target_finish_reason="length"),
            _make_result(sample_id="1"),
        ]
        report = compute_report(results)
        assert report.truncated_count == 1

    def test_both_truncated_counts_once(self) -> None:
        """A sample with both sides truncated counts as one truncated sample."""
        results = [
            _make_result(
                sample_id="0",
                baseline_finish_reason="length",
                target_finish_reason="length",
            ),
        ]
        report = compute_report(results)
        assert report.truncated_count == 1

    def test_multiple_truncated(self) -> None:
        results = [
            _make_result(sample_id="0", baseline_finish_reason="length"),
            _make_result(sample_id="1", target_finish_reason="length"),
            _make_result(sample_id="2"),
        ]
        report = compute_report(results)
        assert report.truncated_count == 2

    def test_none_finish_reason_not_truncated(self) -> None:
        results = [
            _make_result(sample_id="0", baseline_finish_reason=None, target_finish_reason=None),
        ]
        report = compute_report(results)
        assert report.truncated_count == 0


class TestJsonSerialisation:
    """Tests for JSON round-trip with truncation fields."""

    def test_to_json_includes_truncation(self) -> None:
        results = [
            _make_result(sample_id="0", baseline_finish_reason="length"),
        ]
        report = compute_report(results)
        data = json.loads(report.to_json())
        assert data["truncated_count"] == 1
        assert data["results"][0]["baseline_finish_reason"] == "length"
        assert data["results"][0]["target_finish_reason"] == "stop"

    def test_load_report_round_trip(self, tmp_path: Path) -> None:
        results = [
            _make_result(sample_id="0", target_finish_reason="length"),
            _make_result(sample_id="1"),
        ]
        report = compute_report(results)
        path = tmp_path / "report.json"
        path.write_text(report.to_json())

        loaded = load_report(path)
        assert loaded.truncated_count == 1
        assert loaded.results[0].target_finish_reason == "length"
        assert loaded.results[1].target_finish_reason == "stop"

    def test_load_report_backward_compat(self, tmp_path: Path) -> None:
        """Reports without truncation fields load with defaults."""
        data = {
            "schema_version": 1,
            "total_samples": 1,
            "divergent_samples": 0,
            "match_samples": 1,
            "divergence_rate": 0.0,
            "results": [{
                "sample_id": "0",
                "prompt": "p",
                "baseline_output": "a",
                "target_output": "a",
                "exact_match": True,
                "classification": "match",
                "context_length": 1,
            }],
        }
        path = tmp_path / "old_report.json"
        path.write_text(json.dumps(data))
        loaded = load_report(path)
        assert loaded.truncated_count == 0
        assert loaded.results[0].baseline_finish_reason is None
        assert loaded.results[0].target_finish_reason is None


class TestFormatReport:
    """Tests for truncation in formatted output."""

    def test_format_report_shows_truncated(self) -> None:
        results = [
            _make_result(sample_id="0", baseline_finish_reason="length"),
            _make_result(sample_id="1"),
        ]
        report = compute_report(results)
        text = format_report(report)
        assert "Truncated: 1" in text
        assert "⚠️" in text

    def test_format_report_no_truncation_no_line(self) -> None:
        results = [_make_result(sample_id="0")]
        report = compute_report(results)
        text = format_report(report)
        assert "Truncated" not in text


class TestMarkdownExport:
    """Tests for truncation info in Markdown export."""

    def test_markdown_includes_truncation(self) -> None:
        results = [
            _make_result(
                sample_id="0", exact_match=False,
                baseline_finish_reason="length",
            ),
        ]
        report = compute_report(results)
        md = export_markdown(report)
        assert "Truncated samples" in md
        assert "⚠️" in md

    def test_markdown_divergent_sample_truncation_flag(self) -> None:
        results = [
            _make_result(
                sample_id="0", exact_match=False,
                target_finish_reason="length",
            ),
        ]
        report = compute_report(results)
        md = export_markdown(report)
        assert "Truncated output detected" in md


class TestSchemaVersion:
    """Tests for schema version bump."""

    def test_schema_version_is_2(self) -> None:
        from xpyd_acc.batch_compare import REPORT_SCHEMA_VERSION
        assert REPORT_SCHEMA_VERSION == 2

    def test_to_json_has_schema_version_2(self) -> None:
        results = [_make_result()]
        report = compute_report(results)
        data = json.loads(report.to_json())
        assert data["schema_version"] == 2
