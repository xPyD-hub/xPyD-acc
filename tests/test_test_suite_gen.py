"""Tests for test suite generation from batch reports."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from xpyd_acc.batch_compare import BatchReport, SampleResult
from xpyd_acc.test_suite_gen import (
    GenerateSuiteConfig,
    SuiteEntry,
    format_suite_summary,
    generate_suite,
    write_suite,
)


def _make_result(
    sample_id: str,
    prompt: str = "test prompt",
    match: bool = False,
    classification: str = "likely_bug",
    logprob_gap: float | None = 0.5,
    divergence_index: int | None = 5,
    baseline_output: str = "baseline",
    target_output: str = "target",
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        prompt=prompt,
        baseline_output=baseline_output,
        target_output=target_output,
        exact_match=match,
        first_divergence_index=divergence_index,
        baseline_logprob_at_divergence=-0.1 if not match else None,
        target_logprob_at_divergence=-0.6 if not match else None,
        logprob_gap=logprob_gap,
        classification=classification,
        context_length=100,
    )


def _make_report(results: list[SampleResult]) -> BatchReport:
    divergent = [r for r in results if not r.exact_match]
    return BatchReport(
        total_samples=len(results),
        divergent_samples=len(divergent),
        match_samples=len(results) - len(divergent),
        divergence_rate=len(divergent) / len(results) if results else 0.0,
        results=results,
    )


class TestSuiteEntry:
    def test_to_dict_without_expected(self):
        entry = SuiteEntry(id="s1", prompt="hello")
        d = entry.to_dict()
        assert d == {"id": "s1", "prompt": "hello"}
        assert "expected" not in d
        assert "metadata" not in d

    def test_to_dict_with_expected_and_metadata(self):
        entry = SuiteEntry(id="s1", prompt="hello", expected="world",
                           metadata={"classification": "likely_bug"})
        d = entry.to_dict()
        assert d["expected"] == "world"
        assert d["metadata"]["classification"] == "likely_bug"

    def test_to_jsonl_line(self):
        entry = SuiteEntry(id="s1", prompt="hello")
        line = entry.to_jsonl_line()
        parsed = json.loads(line)
        assert parsed["id"] == "s1"
        assert parsed["prompt"] == "hello"


class TestGenerateSuite:
    def test_basic_generation(self):
        results = [
            _make_result("s1", match=True),
            _make_result("s2", match=False),
            _make_result("s3", match=False),
        ]
        report = _make_report(results)
        entries = generate_suite(report)
        assert len(entries) == 2
        assert {e.id for e in entries} == {"s2", "s3"}

    def test_empty_report(self):
        report = _make_report([])
        entries = generate_suite(report)
        assert entries == []

    def test_no_divergent_samples(self):
        results = [_make_result("s1", match=True)]
        report = _make_report(results)
        entries = generate_suite(report)
        assert entries == []

    def test_filter_by_classification(self):
        results = [
            _make_result("s1", match=False, classification="likely_bug"),
            _make_result("s2", match=False, classification="likely_uncertainty"),
            _make_result("s3", match=False, classification="likely_bug"),
        ]
        report = _make_report(results)
        config = GenerateSuiteConfig(classification="likely_bug")
        entries = generate_suite(report, config)
        assert len(entries) == 2
        assert all(e.metadata["classification"] == "likely_bug" for e in entries)

    def test_filter_by_min_logprob_gap(self):
        results = [
            _make_result("s1", match=False, logprob_gap=0.1),
            _make_result("s2", match=False, logprob_gap=0.5),
            _make_result("s3", match=False, logprob_gap=None),
        ]
        report = _make_report(results)
        config = GenerateSuiteConfig(min_logprob_gap=0.3)
        entries = generate_suite(report, config)
        assert len(entries) == 1
        assert entries[0].id == "s2"

    def test_max_samples(self):
        results = [_make_result(f"s{i}", match=False) for i in range(10)]
        report = _make_report(results)
        config = GenerateSuiteConfig(max_samples=3)
        entries = generate_suite(report, config)
        assert len(entries) == 3

    def test_include_expected(self):
        results = [_make_result("s1", match=False, baseline_output="expected output")]
        report = _make_report(results)
        config = GenerateSuiteConfig(include_expected=True)
        entries = generate_suite(report, config)
        assert entries[0].expected == "expected output"

    def test_exclude_expected_by_default(self):
        results = [_make_result("s1", match=False, baseline_output="expected output")]
        report = _make_report(results)
        entries = generate_suite(report)
        assert entries[0].expected is None

    def test_metadata_populated(self):
        results = [_make_result("s1", match=False, classification="likely_bug",
                                logprob_gap=0.42, divergence_index=7)]
        report = _make_report(results)
        entries = generate_suite(report)
        meta = entries[0].metadata
        assert meta["classification"] == "likely_bug"
        assert meta["logprob_gap"] == 0.42
        assert meta["divergence_index"] == 7

    def test_combined_filters(self):
        results = [
            _make_result("s1", match=False, classification="likely_bug", logprob_gap=0.1),
            _make_result("s2", match=False, classification="likely_bug", logprob_gap=0.5),
            _make_result("s3", match=False, classification="likely_uncertainty", logprob_gap=0.8),
        ]
        report = _make_report(results)
        config = GenerateSuiteConfig(classification="likely_bug", min_logprob_gap=0.3)
        entries = generate_suite(report, config)
        assert len(entries) == 1
        assert entries[0].id == "s2"


class TestWriteSuite:
    def test_write_and_read_back(self):
        entries = [
            SuiteEntry(id="s1", prompt="hello", metadata={"classification": "likely_bug"}),
            SuiteEntry(id="s2", prompt="world", expected="output"),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "suite.jsonl"
            write_suite(entries, path)

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2

            parsed = [json.loads(line) for line in lines]
            assert parsed[0]["id"] == "s1"
            assert parsed[0]["prompt"] == "hello"
            assert parsed[1]["expected"] == "output"

    def test_round_trip_with_batch_compare_dataset(self):
        """Generated suite should be loadable as a batch-compare dataset."""
        entries = [
            SuiteEntry(id="s1", prompt="What is 2+2?"),
            SuiteEntry(id="s2", prompt="Hello world"),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "suite.jsonl"
            write_suite(entries, path)

            # Verify JSONL is valid and has required 'prompt' field
            for line in path.read_text().strip().split("\n"):
                data = json.loads(line)
                assert "prompt" in data
                assert "id" in data

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub" / "dir" / "suite.jsonl"
            write_suite([SuiteEntry(id="s1", prompt="test")], path)
            assert path.exists()


class TestFormatSuiteSummary:
    def test_summary_with_entries(self):
        results = [
            _make_result("s1", match=False, classification="likely_bug"),
            _make_result("s2", match=False, classification="likely_uncertainty"),
        ]
        report = _make_report(results)
        entries = generate_suite(report)
        summary = format_suite_summary(entries, report)
        assert "2 samples" in summary
        assert "likely_bug" in summary
        assert "likely_uncertainty" in summary

    def test_summary_empty(self):
        report = _make_report([_make_result("s1", match=True)])
        summary = format_suite_summary([], report)
        assert "No samples matched" in summary
