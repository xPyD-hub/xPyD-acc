"""Tests for report module — HTML generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from xpyd_acc.batch_compare import BatchReport, SampleResult, compute_report
from xpyd_acc.report import generate_html_report, write_html_report


def _make_results() -> list[SampleResult]:
    """Create a small set of sample results for testing."""
    return [
        SampleResult(
            sample_id="0",
            prompt="What is 2+2?",
            baseline_output="4",
            target_output="4",
            exact_match=True,
            first_divergence_index=None,
            baseline_logprob_at_divergence=None,
            target_logprob_at_divergence=None,
            logprob_gap=None,
            classification="match",
            context_length=5,
        ),
        SampleResult(
            sample_id="1",
            prompt="Explain gravity in detail",
            baseline_output="Gravity is a fundamental force",
            target_output="Gravity is a basic force",
            exact_match=False,
            first_divergence_index=3,
            baseline_logprob_at_divergence=-0.5,
            target_logprob_at_divergence=-0.8,
            logprob_gap=0.3,
            classification="likely_bug",
            context_length=25,
        ),
        SampleResult(
            sample_id="2",
            prompt="Write hello world",
            baseline_output="print('hello')",
            target_output="print('Hello')",
            exact_match=False,
            first_divergence_index=0,
            baseline_logprob_at_divergence=-1.0,
            target_logprob_at_divergence=-1.1,
            logprob_gap=0.05,
            classification="likely_uncertainty",
            context_length=15,
        ),
    ]


def _make_report() -> BatchReport:
    """Build a BatchReport from test results."""
    return compute_report(_make_results())


class TestGenerateHtmlReport:
    """Tests for generate_html_report."""

    def test_returns_html_string(self) -> None:
        report = _make_report()
        html = generate_html_report(report)
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")

    def test_contains_summary_dashboard(self) -> None:
        report = _make_report()
        html = generate_html_report(report)
        assert "Total Samples" in html
        assert "Match Rate" in html
        assert "Divergent" in html
        assert "Likely Bugs" in html

    def test_contains_sample_ids(self) -> None:
        report = _make_report()
        html = generate_html_report(report)
        for r in report.results:
            assert r.sample_id in html

    def test_contains_charts(self) -> None:
        report = _make_report()
        html = generate_html_report(report)
        assert "Context Length vs Divergence Rate" in html
        assert "Divergence Point Distribution" in html

    def test_contains_expandable_details(self) -> None:
        report = _make_report()
        html = generate_html_report(report)
        assert "toggleDetail" in html
        assert "detail-0" in html

    def test_heatmap_for_divergent_sample(self) -> None:
        report = _make_report()
        html = generate_html_report(report)
        assert "heatmap-cell" in html

    def test_escapes_html_in_prompts(self) -> None:
        results = [SampleResult(
            sample_id="xss",
            prompt="<script>alert('xss')</script>",
            baseline_output="safe",
            target_output="safe",
            exact_match=True,
            first_divergence_index=None,
            baseline_logprob_at_divergence=None,
            target_logprob_at_divergence=None,
            logprob_gap=None,
            classification="match",
            context_length=5,
        )]
        report = compute_report(results)
        html = generate_html_report(report)
        assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html

    def test_empty_report(self) -> None:
        report = compute_report([])
        html = generate_html_report(report)
        assert "Total Samples" in html
        assert isinstance(html, str)


class TestWriteHtmlReport:
    """Tests for write_html_report."""

    def test_writes_file(self) -> None:
        report = _make_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            write_html_report(report, path)
            assert path.exists()
            content = path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_file_contains_all_sections(self) -> None:
        report = _make_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.html"
            write_html_report(report, path)
            content = path.read_text()
            assert "dashboard" in content
            assert "chart-box" in content
            assert "sample-row" in content


class TestCliReport:
    """Test the CLI report subcommand."""

    def test_report_from_json(self) -> None:
        from xpyd_acc.cli import main

        results = _make_results()
        report = compute_report(results)

        # Serialize report to JSON
        data = {
            "total_samples": report.total_samples,
            "divergent_samples": report.divergent_samples,
            "match_samples": report.match_samples,
            "divergence_rate": report.divergence_rate,
            "divergence_index_mean": report.divergence_index_mean,
            "divergence_index_median": report.divergence_index_median,
            "logprob_gap_mean": report.logprob_gap_mean,
            "logprob_gap_median": report.logprob_gap_median,
            "likely_bugs": report.likely_bugs,
            "likely_uncertainty": report.likely_uncertainty,
            "unknown_classification": report.unknown_classification,
            "divergence_by_context_length": report.divergence_by_context_length,
            "results": [
                {
                    "sample_id": r.sample_id,
                    "prompt": r.prompt,
                    "baseline_output": r.baseline_output,
                    "target_output": r.target_output,
                    "exact_match": r.exact_match,
                    "first_divergence_index": r.first_divergence_index,
                    "baseline_logprob_at_divergence": r.baseline_logprob_at_divergence,
                    "target_logprob_at_divergence": r.target_logprob_at_divergence,
                    "logprob_gap": r.logprob_gap,
                    "classification": r.classification,
                    "context_length": r.context_length,
                }
                for r in results
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "batch.json"
            html_path = Path(tmp) / "report.html"
            json_path.write_text(json.dumps(data))
            main(["report", "--input", str(json_path), "--output", str(html_path)])
            assert html_path.exists()
            content = html_path.read_text()
            assert "<!DOCTYPE html>" in content
