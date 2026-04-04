"""Tests for selective sample rerun (M23)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.rerun import (
    RerunPlan,
    _parse_sample_results,
    load_divergent_samples,
    merge_rerun_results,
)


def _make_report_json(results: list[dict], total: int | None = None) -> str:
    """Helper to create a report JSON string."""
    if total is None:
        total = len(results)
    data = {
        "total_samples": total,
        "divergent_samples": sum(1 for r in results if not r.get("exact_match", True)),
        "match_samples": sum(1 for r in results if r.get("exact_match", True)),
        "divergence_rate": 0.0,
        "divergence_index_mean": None,
        "divergence_index_median": None,
        "logprob_gap_mean": None,
        "logprob_gap_median": None,
        "likely_bugs": 0,
        "likely_uncertainty": 0,
        "unknown_classification": 0,
        "divergence_by_context_length": {},
        "results": results,
    }
    return json.dumps(data)


def _sample_result(sample_id: str, match: bool, prompt: str = "test") -> dict:
    """Create a sample result dict."""
    return {
        "sample_id": sample_id,
        "prompt": prompt,
        "baseline_output": "hello",
        "target_output": "hello" if match else "world",
        "exact_match": match,
        "first_divergence_index": None if match else 0,
        "baseline_logprob_at_divergence": None,
        "target_logprob_at_divergence": None,
        "logprob_gap": None,
        "classification": "match" if match else "likely_bug",
        "context_length": 5,
    }


class TestLoadDivergentSamples:
    """Tests for load_divergent_samples."""

    def test_basic_load(self, tmp_path: Path) -> None:
        results = [
            _sample_result("s1", match=True),
            _sample_result("s2", match=False),
            _sample_result("s3", match=False, prompt="other"),
        ]
        report = tmp_path / "report.json"
        report.write_text(_make_report_json(results))

        plan = load_divergent_samples(report)
        assert isinstance(plan, RerunPlan)
        assert plan.divergent_count == 2
        assert plan.total_in_report == 3
        assert len(plan.divergent_samples) == 2
        ids = {s.id for s in plan.divergent_samples}
        assert ids == {"s2", "s3"}

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_divergent_samples("/nonexistent/report.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        report = tmp_path / "bad.json"
        report.write_text("not json at all")
        with pytest.raises(json.JSONDecodeError):
            load_divergent_samples(report)

    def test_missing_results_key(self, tmp_path: Path) -> None:
        report = tmp_path / "report.json"
        report.write_text(json.dumps({"total_samples": 5}))
        with pytest.raises(ValueError, match="results"):
            load_divergent_samples(report)

    def test_no_divergent_samples(self, tmp_path: Path) -> None:
        results = [_sample_result("s1", match=True), _sample_result("s2", match=True)]
        report = tmp_path / "report.json"
        report.write_text(_make_report_json(results))
        with pytest.raises(ValueError, match="No divergent"):
            load_divergent_samples(report)

    def test_all_divergent(self, tmp_path: Path) -> None:
        results = [_sample_result("s1", match=False), _sample_result("s2", match=False)]
        report = tmp_path / "report.json"
        report.write_text(_make_report_json(results))
        plan = load_divergent_samples(report)
        assert plan.divergent_count == 2
        assert plan.total_in_report == 2

    def test_preserves_prompt(self, tmp_path: Path) -> None:
        results = [_sample_result("s1", match=False, prompt="my special prompt")]
        report = tmp_path / "report.json"
        report.write_text(_make_report_json(results))
        plan = load_divergent_samples(report)
        assert plan.divergent_samples[0].prompt == "my special prompt"


class TestMergeRerunResults:
    """Tests for merge_rerun_results."""

    def test_basic_merge(self, tmp_path: Path) -> None:
        # Original: s1 match, s2 diverges, s3 match
        original_results = [
            _sample_result("s1", match=True),
            _sample_result("s2", match=False),
            _sample_result("s3", match=True),
        ]
        report_path = tmp_path / "report.json"
        report_path.write_text(_make_report_json(original_results))

        # Rerun: s2 now matches
        from xpyd_acc.batch_compare import SampleResult, compute_report

        rerun_result = SampleResult(
            sample_id="s2",
            prompt="test",
            baseline_output="hello",
            target_output="hello",
            exact_match=True,
            first_divergence_index=None,
            baseline_logprob_at_divergence=None,
            target_logprob_at_divergence=None,
            logprob_gap=None,
            classification="match",
            context_length=5,
        )
        rerun_report = compute_report([rerun_result])

        merged = merge_rerun_results(report_path, rerun_report)
        assert merged.total_samples == 3
        assert merged.match_samples == 3
        assert merged.divergent_samples == 0

    def test_merge_still_divergent(self, tmp_path: Path) -> None:
        original_results = [
            _sample_result("s1", match=True),
            _sample_result("s2", match=False),
        ]
        report_path = tmp_path / "report.json"
        report_path.write_text(_make_report_json(original_results))

        from xpyd_acc.batch_compare import SampleResult, compute_report

        rerun_result = SampleResult(
            sample_id="s2",
            prompt="test",
            baseline_output="hello",
            target_output="different",
            exact_match=False,
            first_divergence_index=0,
            baseline_logprob_at_divergence=None,
            target_logprob_at_divergence=None,
            logprob_gap=None,
            classification="likely_bug",
            context_length=5,
        )
        rerun_report = compute_report([rerun_result])

        merged = merge_rerun_results(report_path, rerun_report)
        assert merged.total_samples == 2
        assert merged.divergent_samples == 1


class TestParseSampleResults:
    """Tests for _parse_sample_results."""

    def test_parses_correctly(self) -> None:
        data = [_sample_result("s1", match=True), _sample_result("s2", match=False)]
        parsed = _parse_sample_results(data)
        assert len(parsed) == 2
        assert parsed[0].sample_id == "s1"
        assert parsed[0].exact_match is True
        assert parsed[1].sample_id == "s2"
        assert parsed[1].exact_match is False

    def test_handles_missing_optional_fields(self) -> None:
        data = [{
            "sample_id": "s1",
            "prompt": "test",
            "baseline_output": "a",
            "target_output": "b",
            "exact_match": False,
        }]
        parsed = _parse_sample_results(data)
        assert parsed[0].classification == "unknown"
        assert parsed[0].context_length == 0
