"""Tests for multi-run aggregation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from xpyd_acc.aggregate import (
    aggregate_reports,
    format_aggregated_report,
    load_batch_report_from_json,
)
from xpyd_acc.batch_compare import BatchReport, SampleResult, compute_report


def _make_result(
    sample_id: str,
    match: bool,
    prompt: str = "test prompt",
) -> SampleResult:
    """Create a minimal SampleResult for testing."""
    return SampleResult(
        sample_id=sample_id,
        prompt=prompt,
        baseline_output="hello world",
        target_output="hello world" if match else "hello mars",
        exact_match=match,
        first_divergence_index=None if match else 1,
        baseline_logprob_at_divergence=None,
        target_logprob_at_divergence=None,
        logprob_gap=None if match else 0.5,
        classification="match" if match else "likely_bug",
        context_length=2,
    )


def _make_report(results: list[SampleResult]) -> BatchReport:
    """Build a BatchReport from a list of SampleResults."""
    return compute_report(results)


class TestAggregateReports:
    """Test aggregate_reports function."""

    def test_all_stable(self) -> None:
        """All samples match in all runs → all stable."""
        r1 = _make_report([_make_result("s1", True), _make_result("s2", True)])
        r2 = _make_report([_make_result("s1", True), _make_result("s2", True)])

        agg = aggregate_reports([r1, r2])
        assert agg.total_runs == 2
        assert agg.total_unique_samples == 2
        assert agg.stable_count == 2
        assert agg.persistent_count == 0
        assert agg.flaky_count == 0
        assert agg.stable_rate == 1.0

    def test_all_persistent(self) -> None:
        """All samples diverge in all runs → all persistent."""
        r1 = _make_report([_make_result("s1", False), _make_result("s2", False)])
        r2 = _make_report([_make_result("s1", False), _make_result("s2", False)])

        agg = aggregate_reports([r1, r2])
        assert agg.persistent_count == 2
        assert agg.flaky_count == 0
        assert agg.stable_count == 0
        assert agg.persistent_rate == 1.0

    def test_mixed_flaky(self) -> None:
        """Sample diverges in some runs but not others → flaky."""
        r1 = _make_report([_make_result("s1", True), _make_result("s2", False)])
        r2 = _make_report([_make_result("s1", False), _make_result("s2", False)])

        agg = aggregate_reports([r1, r2])
        samples_by_id = {s.sample_id: s for s in agg.samples}

        assert samples_by_id["s1"].classification == "flaky"
        assert samples_by_id["s1"].consistency_score == 0.5
        assert samples_by_id["s2"].classification == "persistent"
        assert samples_by_id["s2"].consistency_score == 1.0

    def test_single_run(self) -> None:
        """Single run edge case."""
        r1 = _make_report([_make_result("s1", True), _make_result("s2", False)])

        agg = aggregate_reports([r1])
        assert agg.total_runs == 1
        samples_by_id = {s.sample_id: s for s in agg.samples}
        assert samples_by_id["s1"].classification == "stable"
        assert samples_by_id["s2"].classification == "persistent"

    def test_empty_raises(self) -> None:
        """No reports raises ValueError."""
        with pytest.raises(ValueError, match="At least one report"):
            aggregate_reports([])

    def test_consistency_score(self) -> None:
        """Consistency score = diverge_count / run_count."""
        r1 = _make_report([_make_result("s1", False)])
        r2 = _make_report([_make_result("s1", True)])
        r3 = _make_report([_make_result("s1", False)])

        agg = aggregate_reports([r1, r2, r3])
        s = agg.samples[0]
        assert s.diverge_count == 2
        assert s.match_count == 1
        assert abs(s.consistency_score - 2 / 3) < 1e-9

    def test_json_export(self) -> None:
        """to_json() produces valid JSON with expected fields."""
        r1 = _make_report([_make_result("s1", True), _make_result("s2", False)])
        r2 = _make_report([_make_result("s1", False), _make_result("s2", False)])

        agg = aggregate_reports([r1, r2])
        data = json.loads(agg.to_json())

        assert data["total_runs"] == 2
        assert data["total_unique_samples"] == 2
        assert len(data["samples"]) == 2
        assert "consistency_score" in data["samples"][0]

    def test_format_report(self) -> None:
        """format_aggregated_report produces human-readable output."""
        r1 = _make_report([_make_result("s1", True), _make_result("s2", False)])
        r2 = _make_report([_make_result("s1", False), _make_result("s2", False)])

        agg = aggregate_reports([r1, r2])
        text = format_aggregated_report(agg)

        assert "Multi-Run Aggregation Report" in text
        assert "Persistent" in text
        assert "Flaky" in text


class TestLoadBatchReportFromJson:
    """Test loading BatchReport from JSON files."""

    def test_round_trip(self) -> None:
        """Save a report as JSON, load it back, verify results."""
        original = _make_report([
            _make_result("s1", True),
            _make_result("s2", False),
        ])
        json_str = original.to_json()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_str)
            path = f.name

        loaded = load_batch_report_from_json(path)
        Path(path).unlink()

        assert loaded.total_samples == 2
        assert len(loaded.results) == 2


class TestCliAggregate:
    """Test CLI aggregate subcommand integration."""

    def test_aggregate_cli(self) -> None:
        """CLI aggregate reads report files and prints output."""
        from xpyd_acc.cli import main

        r1 = _make_report([_make_result("s1", True), _make_result("s2", False)])
        r2 = _make_report([_make_result("s1", False), _make_result("s2", False)])

        paths = []
        for r in [r1, r2]:
            f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            f.write(r.to_json())
            f.close()
            paths.append(f.name)

        try:
            main(["aggregate", "--reports"] + paths)
        finally:
            for p in paths:
                Path(p).unlink()

    def test_aggregate_json_export(self, tmp_path: Path) -> None:
        """CLI aggregate --json exports file."""
        from xpyd_acc.cli import main

        r1 = _make_report([_make_result("s1", True)])
        r2 = _make_report([_make_result("s1", True)])

        paths = []
        for r in [r1, r2]:
            f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            f.write(r.to_json())
            f.close()
            paths.append(f.name)

        out = tmp_path / "agg.json"
        try:
            main(["aggregate", "--reports"] + paths + ["--json", str(out)])
        finally:
            for p in paths:
                Path(p).unlink()

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total_runs"] == 2
        assert data["stable_count"] == 1
