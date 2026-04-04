"""Tests for regression detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.regression import (
    RegressionReport,
    SampleDelta,
    compare_runs,
    format_regression_report,
)


def _make_result(sample_id: str, prompt: str, match: bool) -> dict:
    """Create a minimal batch result entry."""
    return {
        "sample_id": sample_id,
        "prompt": prompt,
        "baseline_output": "output_a",
        "target_output": "output_a" if match else "output_b",
        "exact_match": match,
        "first_divergence_index": None if match else 5,
        "logprob_gap": None if match else 0.3,
        "classification": "match" if match else "likely_bug",
        "context_length": 100,
    }


def _write_batch_json(path: Path, results: list[dict]) -> None:
    """Write batch results as JSON."""
    data = {
        "total_samples": len(results),
        "divergent_samples": sum(1 for r in results if not r["exact_match"]),
        "match_samples": sum(1 for r in results if r["exact_match"]),
        "divergence_rate": sum(1 for r in results if not r["exact_match"]) / max(len(results), 1),
        "results": results,
    }
    path.write_text(json.dumps(data))


class TestCompareRuns:
    """Tests for compare_runs function."""

    def test_no_regressions(self, tmp_path: Path) -> None:
        baseline = [_make_result("s1", "hello", True), _make_result("s2", "world", False)]
        current = [_make_result("s1", "hello", True), _make_result("s2", "world", True)]

        _write_batch_json(tmp_path / "baseline.json", baseline)
        _write_batch_json(tmp_path / "current.json", current)

        report = compare_runs(tmp_path / "baseline.json", tmp_path / "current.json")
        assert report.regressions == 0
        assert report.fixes == 1
        assert report.persistent_matches == 1
        assert not report.has_regressions
        assert report.net_change == 1

    def test_with_regressions(self, tmp_path: Path) -> None:
        baseline = [_make_result("s1", "hello", True), _make_result("s2", "world", True)]
        current = [_make_result("s1", "hello", False), _make_result("s2", "world", True)]

        _write_batch_json(tmp_path / "baseline.json", baseline)
        _write_batch_json(tmp_path / "current.json", current)

        report = compare_runs(tmp_path / "baseline.json", tmp_path / "current.json")
        assert report.regressions == 1
        assert report.has_regressions
        assert report.net_change == -1

    def test_persistent_divergences(self, tmp_path: Path) -> None:
        baseline = [_make_result("s1", "hello", False)]
        current = [_make_result("s1", "hello", False)]

        _write_batch_json(tmp_path / "baseline.json", baseline)
        _write_batch_json(tmp_path / "current.json", current)

        report = compare_runs(tmp_path / "baseline.json", tmp_path / "current.json")
        assert report.persistent_divergences == 1
        assert report.regressions == 0

    def test_empty_intersection(self, tmp_path: Path) -> None:
        baseline = [_make_result("s1", "hello", True)]
        current = [_make_result("s2", "world", True)]

        _write_batch_json(tmp_path / "baseline.json", baseline)
        _write_batch_json(tmp_path / "current.json", current)

        report = compare_runs(tmp_path / "baseline.json", tmp_path / "current.json")
        assert report.total_samples == 0

    def test_raw_list_format(self, tmp_path: Path) -> None:
        """Support raw list of results (not wrapped in BatchReport)."""
        baseline = [_make_result("s1", "hello", True)]
        current = [_make_result("s1", "hello", False)]

        (tmp_path / "baseline.json").write_text(json.dumps(baseline))
        (tmp_path / "current.json").write_text(json.dumps(current))

        report = compare_runs(tmp_path / "baseline.json", tmp_path / "current.json")
        assert report.regressions == 1

    def test_file_not_found(self, tmp_path: Path) -> None:
        _write_batch_json(tmp_path / "baseline.json", [])
        with pytest.raises(FileNotFoundError):
            compare_runs(tmp_path / "baseline.json", tmp_path / "missing.json")

    def test_mixed_changes(self, tmp_path: Path) -> None:
        baseline = [
            _make_result("s1", "a", True),
            _make_result("s2", "b", False),
            _make_result("s3", "c", True),
            _make_result("s4", "d", False),
        ]
        current = [
            _make_result("s1", "a", False),  # regression
            _make_result("s2", "b", True),   # fix
            _make_result("s3", "c", True),   # persistent match
            _make_result("s4", "d", False),  # persistent divergence
        ]

        _write_batch_json(tmp_path / "baseline.json", baseline)
        _write_batch_json(tmp_path / "current.json", current)

        report = compare_runs(tmp_path / "baseline.json", tmp_path / "current.json")
        assert report.regressions == 1
        assert report.fixes == 1
        assert report.persistent_matches == 1
        assert report.persistent_divergences == 1
        assert report.net_change == 0
        assert report.total_samples == 4

    def test_divergence_rates(self, tmp_path: Path) -> None:
        baseline = [_make_result("s1", "a", True), _make_result("s2", "b", False)]
        current = [_make_result("s1", "a", False), _make_result("s2", "b", False)]

        _write_batch_json(tmp_path / "baseline.json", baseline)
        _write_batch_json(tmp_path / "current.json", current)

        report = compare_runs(tmp_path / "baseline.json", tmp_path / "current.json")
        assert report.baseline_divergence_rate == 0.5
        assert report.current_divergence_rate == 1.0


class TestRegressionReport:
    """Tests for RegressionReport."""

    def test_to_json(self) -> None:
        report = RegressionReport(
            total_samples=10,
            regressions=2,
            fixes=3,
            persistent_divergences=1,
            persistent_matches=4,
            net_change=1,
            baseline_divergence_rate=0.3,
            current_divergence_rate=0.3,
        )
        data = json.loads(report.to_json())
        assert data["total_samples"] == 10
        assert data["regressions"] == 2
        assert data["has_regressions"] is True

    def test_no_regressions_json(self) -> None:
        report = RegressionReport(
            total_samples=5,
            regressions=0,
            fixes=1,
            persistent_divergences=0,
            persistent_matches=4,
            net_change=1,
            baseline_divergence_rate=0.2,
            current_divergence_rate=0.0,
        )
        data = json.loads(report.to_json())
        assert data["has_regressions"] is False


class TestSampleDelta:
    """Tests for SampleDelta."""

    def test_is_regression(self) -> None:
        d = SampleDelta("s1", "regression", True, False, "hello")
        assert d.is_regression()
        assert not d.is_fix()

    def test_is_fix(self) -> None:
        d = SampleDelta("s1", "fix", False, True, "hello")
        assert d.is_fix()
        assert not d.is_regression()


class TestFormatReport:
    """Tests for format_regression_report."""

    def test_pass_output(self) -> None:
        report = RegressionReport(
            total_samples=5,
            regressions=0,
            fixes=2,
            persistent_divergences=0,
            persistent_matches=3,
            net_change=2,
            baseline_divergence_rate=0.4,
            current_divergence_rate=0.0,
        )
        output = format_regression_report(report)
        assert "NO REGRESSIONS" in output
        assert "Fixes:" not in output or "2" in output

    def test_fail_output(self) -> None:
        report = RegressionReport(
            total_samples=5,
            regressions=1,
            fixes=0,
            persistent_divergences=1,
            persistent_matches=3,
            net_change=-1,
            baseline_divergence_rate=0.2,
            current_divergence_rate=0.4,
            deltas=[
                SampleDelta("s1", "regression", True, False, "test prompt"),
            ],
        )
        output = format_regression_report(report)
        assert "REGRESSIONS FOUND" in output
        assert "s1" in output
