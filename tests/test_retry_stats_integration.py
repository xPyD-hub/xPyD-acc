"""Tests for M66: Retry Statistics Reporting integration."""

from __future__ import annotations

import json

from xpyd_acc.batch_compare import (
    SampleResult,
    compute_report,
    format_report,
    load_report,
)
from xpyd_acc.retry import RetryStats


def _make_sample(sid: str, match: bool) -> SampleResult:
    return SampleResult(
        sample_id=sid,
        prompt="test",
        baseline_output="a",
        target_output="a" if match else "b",
        exact_match=match,
        first_divergence_index=None if match else 0,
        baseline_logprob_at_divergence=None,
        target_logprob_at_divergence=None,
        logprob_gap=None if match else 0.5,
        classification="match" if match else "likely_bug",
        context_length=10,
    )


class TestRetryStatsInReport:
    def test_report_includes_retry_stats_in_json(self) -> None:
        report = compute_report([_make_sample("s1", True)])
        report.retry_stats = RetryStats(
            total_requests=4, total_retries=2,
            max_retries_single=2, retried_request_count=1,
        )
        data = json.loads(report.to_json())
        assert data["retry_stats"] is not None
        assert data["retry_stats"]["total_requests"] == 4
        assert data["retry_stats"]["total_retries"] == 2
        assert data["retry_stats"]["max_retries_single"] == 2
        assert data["retry_stats"]["retried_request_count"] == 1

    def test_report_no_retry_stats_json(self) -> None:
        report = compute_report([_make_sample("s1", True)])
        data = json.loads(report.to_json())
        assert data["retry_stats"] is None

    def test_report_round_trip_with_retry_stats(self, tmp_path) -> None:
        report = compute_report([_make_sample("s1", True)])
        report.retry_stats = RetryStats(
            total_requests=6, total_retries=3,
            max_retries_single=2, retried_request_count=2,
        )
        path = tmp_path / "report.json"
        path.write_text(report.to_json())
        loaded = load_report(str(path))
        assert loaded.retry_stats is not None
        assert loaded.retry_stats.total_requests == 6
        assert loaded.retry_stats.total_retries == 3
        assert loaded.retry_stats.max_retries_single == 2
        assert loaded.retry_stats.retried_request_count == 2

    def test_load_report_without_retry_stats_backward_compat(self, tmp_path) -> None:
        report = compute_report([_make_sample("s1", True)])
        data = json.loads(report.to_json())
        del data["retry_stats"]
        path = tmp_path / "report.json"
        path.write_text(json.dumps(data))
        loaded = load_report(str(path))
        assert loaded.retry_stats is None

    def test_format_report_with_retries(self) -> None:
        report = compute_report([_make_sample("s1", True)])
        report.retry_stats = RetryStats(
            total_requests=10, total_retries=5,
            max_retries_single=3, retried_request_count=3,
        )
        text = format_report(report)
        assert "Retry Statistics" in text
        assert "Total requests:     10" in text
        assert "Total retries:      5" in text

    def test_format_report_no_retries_suppressed(self) -> None:
        report = compute_report([_make_sample("s1", True)])
        report.retry_stats = RetryStats(total_requests=4, total_retries=0)
        text = format_report(report)
        assert "Retry Statistics" not in text

    def test_format_report_none_retry_stats(self) -> None:
        report = compute_report([_make_sample("s1", True)])
        text = format_report(report)
        assert "Retry Statistics" not in text

    def test_markdown_includes_retry_stats(self) -> None:
        report = compute_report([_make_sample("s1", True)])
        report.retry_stats = RetryStats(
            total_requests=8, total_retries=3,
            max_retries_single=2, retried_request_count=2,
        )
        md = report.to_markdown()
        assert "## Retry Statistics" in md
        assert "Total retries | 3" in md

    def test_markdown_no_retries_suppressed(self) -> None:
        report = compute_report([_make_sample("s1", True)])
        report.retry_stats = RetryStats(total_requests=4, total_retries=0)
        md = report.to_markdown()
        assert "Retry Statistics" not in md
