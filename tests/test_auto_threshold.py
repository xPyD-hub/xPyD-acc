"""Tests for auto_threshold module."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_acc.auto_threshold import (
    ThresholdRecommendation,
    _percentile,
    analyze_thresholds,
    format_recommendations,
    load_reports,
)
from xpyd_acc.batch_compare import BatchReport, SampleResult


def _make_result(
    match: bool = True,
    logprob_gap: float | None = None,
    sample_id: str = "s1",
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        prompt="test",
        baseline_output="a",
        target_output="a" if match else "b",
        exact_match=match,
        first_divergence_index=None if match else 5,
        baseline_logprob_at_divergence=None,
        target_logprob_at_divergence=None,
        logprob_gap=logprob_gap,
        classification="match" if match else "likely_bug",
        context_length=10,
    )


def _make_report(
    divergence_rate: float,
    total: int = 100,
    logprob_gaps: list[float] | None = None,
) -> BatchReport:
    divergent = int(total * divergence_rate)
    match = total - divergent
    results: list[SampleResult] = []
    gaps = logprob_gaps or []
    for i in range(divergent):
        gap = gaps[i] if i < len(gaps) else 0.05
        results.append(_make_result(match=False, logprob_gap=gap, sample_id=f"d{i}"))
    for i in range(match):
        results.append(_make_result(match=True, sample_id=f"m{i}"))
    return BatchReport(
        total_samples=total,
        divergent_samples=divergent,
        match_samples=match,
        divergence_rate=divergence_rate,
        results=results,
    )


class TestPercentile:
    def test_basic(self):
        assert _percentile([1, 2, 3, 4, 5], 0.5) == 3.0

    def test_p95(self):
        data = list(range(1, 101))
        p = _percentile(data, 0.95)
        assert 95 <= p <= 96

    def test_empty(self):
        assert _percentile([], 0.5) == 0.0

    def test_single(self):
        assert _percentile([42], 0.5) == 42.0


class TestAnalyzeThresholds:
    def test_empty_reports(self):
        rec = analyze_thresholds([])
        assert rec.fail_threshold is None
        assert rec.confidence == "low"
        assert rec.sample_size == 0

    def test_single_report(self):
        report = _make_report(0.1, total=50, logprob_gaps=[0.02, 0.04, 0.06, 0.08, 0.1])
        rec = analyze_thresholds([report])
        assert rec.fail_threshold is not None
        assert rec.fail_threshold > 0.1
        assert rec.numeric_tolerance is not None
        assert rec.confidence == "low"  # only 50 samples, 1 report
        assert rec.sample_size == 50

    def test_multiple_reports_high_confidence(self):
        reports = [
            _make_report(0.05, total=200, logprob_gaps=[0.01] * 10),
            _make_report(0.08, total=200, logprob_gaps=[0.02] * 16),
            _make_report(0.03, total=200, logprob_gaps=[0.005] * 6),
        ]
        rec = analyze_thresholds(reports)
        assert rec.confidence == "high"
        assert rec.sample_size == 600
        assert rec.fail_threshold is not None
        assert len(rec.divergence_rates) == 3

    def test_no_logprob_gaps(self):
        report = _make_report(0.0, total=100)
        rec = analyze_thresholds([report])
        assert rec.numeric_tolerance is None
        assert "cannot recommend numeric_tolerance" in rec.reasoning[-2].lower() or \
               any("cannot recommend" in r.lower() for r in rec.reasoning)

    def test_custom_percentile(self):
        reports = [_make_report(0.1, total=100, logprob_gaps=[0.05] * 10)]
        rec_90 = analyze_thresholds(reports, percentile_level=0.90)
        rec_99 = analyze_thresholds(reports, percentile_level=0.99)
        assert rec_90.percentile_used == 0.90
        assert rec_99.percentile_used == 0.99


class TestThresholdRecommendation:
    def test_to_dict(self):
        rec = ThresholdRecommendation(
            fail_threshold=0.1,
            numeric_tolerance=0.05,
            confidence="high",
            sample_size=500,
            reasoning=["test"],
            divergence_rates=[0.05, 0.1],
            logprob_gaps=[0.01, 0.02],
            percentile_used=0.95,
        )
        d = rec.to_dict()
        assert d["fail_threshold"] == 0.1
        assert d["numeric_tolerance"] == 0.05
        assert d["confidence"] == "high"
        assert d["sample_size"] == 500
        assert len(d["divergence_rates"]) == 2

    def test_json_serializable(self):
        rec = ThresholdRecommendation(
            fail_threshold=0.1,
            numeric_tolerance=None,
            confidence="low",
            sample_size=10,
            reasoning=["r1"],
        )
        s = json.dumps(rec.to_dict())
        loaded = json.loads(s)
        assert loaded["fail_threshold"] == 0.1


class TestFormatRecommendations:
    def test_format_with_data(self):
        rec = ThresholdRecommendation(
            fail_threshold=0.1,
            numeric_tolerance=0.05,
            confidence="medium",
            sample_size=200,
            reasoning=["reason 1", "reason 2"],
            divergence_rates=[0.05, 0.08],
            logprob_gaps=[0.01] * 5,
        )
        out = format_recommendations(rec)
        assert "0.100" in out
        assert "0.0500" in out
        assert "medium" in out
        assert "reason 1" in out

    def test_format_no_data(self):
        rec = ThresholdRecommendation(
            fail_threshold=None,
            numeric_tolerance=None,
            confidence="low",
            sample_size=0,
            reasoning=["No reports"],
        )
        out = format_recommendations(rec)
        assert "(no data)" in out


class TestLoadReports:
    def test_load_valid(self, tmp_path: Path):
        report = _make_report(0.1, total=10)
        p = tmp_path / "r.json"
        p.write_text(report.to_json(), encoding="utf-8")
        loaded = load_reports([str(p)])
        assert len(loaded) == 1
        assert loaded[0].total_samples == 10

    def test_load_missing_file(self):
        loaded = load_reports(["/nonexistent/report.json"])
        assert len(loaded) == 0

    def test_load_mixed(self, tmp_path: Path):
        report = _make_report(0.05, total=20)
        p = tmp_path / "good.json"
        p.write_text(report.to_json(), encoding="utf-8")
        loaded = load_reports([str(p), "/bad/path.json"])
        assert len(loaded) == 1


class TestCLIIntegration:
    def test_auto_threshold_cli(self, tmp_path: Path):
        from xpyd_acc.cli import main

        report = _make_report(0.1, total=50, logprob_gaps=[0.03, 0.05, 0.07, 0.04, 0.06])
        rp = tmp_path / "report.json"
        rp.write_text(report.to_json(), encoding="utf-8")
        jp = tmp_path / "out.json"

        main(["auto-threshold", "--reports", str(rp), "--json", str(jp)])

        assert jp.exists()
        data = json.loads(jp.read_text(encoding="utf-8"))
        assert "fail_threshold" in data
        assert "numeric_tolerance" in data

    def test_auto_threshold_custom_percentile(self, tmp_path: Path):
        from xpyd_acc.cli import main

        report = _make_report(0.2, total=30, logprob_gaps=[0.1] * 6)
        rp = tmp_path / "report.json"
        rp.write_text(report.to_json(), encoding="utf-8")
        jp = tmp_path / "out.json"

        main([
            "auto-threshold", "--reports", str(rp),
            "--percentile", "0.90", "--json", str(jp),
        ])

        data = json.loads(jp.read_text(encoding="utf-8"))
        assert data["percentile_used"] == 0.90
