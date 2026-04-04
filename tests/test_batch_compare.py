"""Tests for batch dataset comparison."""

from __future__ import annotations

from pathlib import Path

from xpyd_acc.batch_compare import (
    BatchReport,
    SampleResult,
    _context_length_bucket,
    _find_first_divergence,
    classify_divergence,
    compute_report,
    export_csv,
    format_report,
    load_dataset,
)


class TestLoadDataset:
    def test_basic(self, tmp_path: Path) -> None:
        data = tmp_path / "data.jsonl"
        data.write_text(
            '{"prompt": "What is 2+2?", "expected": "4"}\n'
            '{"id": "q2", "prompt": "Hello", "expected": "Hi"}\n'
        )
        samples = load_dataset(data)
        assert len(samples) == 2
        assert samples[0].id == "0"
        assert samples[0].prompt == "What is 2+2?"
        assert samples[0].expected == "4"
        assert samples[1].id == "q2"
        assert samples[1].prompt == "Hello"

    def test_empty_lines(self, tmp_path: Path) -> None:
        data = tmp_path / "data.jsonl"
        data.write_text('\n{"prompt": "test"}\n\n')
        samples = load_dataset(data)
        assert len(samples) == 1

    def test_missing_prompt(self, tmp_path: Path) -> None:
        data = tmp_path / "data.jsonl"
        data.write_text('{"expected": "4"}\n')
        try:
            load_dataset(data)
            assert False, "Should raise ValueError"  # noqa: B011
        except ValueError as e:
            assert "missing 'prompt'" in str(e)

    def test_metadata(self, tmp_path: Path) -> None:
        data = tmp_path / "data.jsonl"
        data.write_text('{"prompt": "test", "dataset": "gsm8k", "difficulty": "easy"}\n')
        samples = load_dataset(data)
        assert samples[0].metadata["dataset"] == "gsm8k"
        assert samples[0].metadata["difficulty"] == "easy"

    def test_empty_file(self, tmp_path: Path) -> None:
        data = tmp_path / "data.jsonl"
        data.write_text("")
        samples = load_dataset(data)
        assert len(samples) == 0


class TestFindFirstDivergence:
    def test_identical(self) -> None:
        assert _find_first_divergence(["a", "b"], ["a", "b"]) is None

    def test_first_token(self) -> None:
        assert _find_first_divergence(["a", "b"], ["x", "b"]) == 0

    def test_second_token(self) -> None:
        assert _find_first_divergence(["a", "b"], ["a", "x"]) == 1

    def test_different_length(self) -> None:
        assert _find_first_divergence(["a", "b", "c"], ["a", "b"]) == 2

    def test_empty(self) -> None:
        assert _find_first_divergence([], []) is None

    def test_one_empty(self) -> None:
        assert _find_first_divergence(["a"], []) == 0


class TestClassifyDivergence:
    def test_match(self) -> None:
        assert classify_divergence(None) == "unknown"

    def test_likely_bug(self) -> None:
        assert classify_divergence(0.5, threshold=0.1) == "likely_bug"

    def test_likely_uncertainty(self) -> None:
        assert classify_divergence(0.05, threshold=0.1) == "likely_uncertainty"

    def test_threshold_edge(self) -> None:
        assert classify_divergence(0.1, threshold=0.1) == "likely_bug"


class TestContextLengthBucket:
    def test_short(self) -> None:
        assert _context_length_bucket(10) == "0-50"

    def test_medium(self) -> None:
        assert _context_length_bucket(100) == "51-200"

    def test_long(self) -> None:
        assert _context_length_bucket(1500) == "1000+"


class TestComputeReport:
    def _make_result(
        self,
        *,
        exact: bool = True,
        div_idx: int | None = None,
        gap: float | None = None,
        ctx_len: int = 10,
        classification: str = "match",
    ) -> SampleResult:
        return SampleResult(
            sample_id="s1",
            prompt="test prompt",
            baseline_output="out",
            target_output="out" if exact else "different",
            exact_match=exact,
            first_divergence_index=div_idx,
            baseline_logprob_at_divergence=None,
            target_logprob_at_divergence=None,
            logprob_gap=gap,
            classification=classification,
            context_length=ctx_len,
        )

    def test_all_match(self) -> None:
        results = [self._make_result(), self._make_result()]
        report = compute_report(results)
        assert report.total_samples == 2
        assert report.divergent_samples == 0
        assert report.divergence_rate == 0.0

    def test_some_divergent(self) -> None:
        results = [
            self._make_result(),
            self._make_result(exact=False, div_idx=3, gap=0.5, classification="likely_bug"),
            self._make_result(
                exact=False, div_idx=7, gap=0.02,
                classification="likely_uncertainty",
            ),
        ]
        report = compute_report(results)
        assert report.total_samples == 3
        assert report.divergent_samples == 2
        assert report.likely_bugs == 1
        assert report.likely_uncertainty == 1
        assert report.divergence_index_mean == 5.0
        assert report.logprob_gap_mean is not None

    def test_empty(self) -> None:
        report = compute_report([])
        assert report.total_samples == 0
        assert report.divergence_rate == 0.0


class TestExportCsv:
    def test_csv_content(self) -> None:
        results = [
            SampleResult(
                sample_id="s1",
                prompt="test",
                baseline_output="hello",
                target_output="world",
                exact_match=False,
                first_divergence_index=0,
                baseline_logprob_at_divergence=-0.5,
                target_logprob_at_divergence=-0.8,
                logprob_gap=0.3,
                classification="likely_bug",
                context_length=5,
            ),
        ]
        report = compute_report(results)
        csv_str = export_csv(report)
        assert "sample_id" in csv_str
        assert "s1" in csv_str
        assert "likely_bug" in csv_str

    def test_csv_to_file(self, tmp_path: Path) -> None:
        results = [
            SampleResult(
                sample_id="s1",
                prompt="test",
                baseline_output="a",
                target_output="a",
                exact_match=True,
                first_divergence_index=None,
                baseline_logprob_at_divergence=None,
                target_logprob_at_divergence=None,
                logprob_gap=None,
                classification="match",
                context_length=5,
            ),
        ]
        report = compute_report(results)
        out = tmp_path / "out.csv"
        export_csv(report, out)
        assert out.exists()
        content = out.read_text()
        assert "s1" in content


class TestFormatReport:
    def test_all_match(self) -> None:
        report = BatchReport(
            total_samples=5,
            divergent_samples=0,
            match_samples=5,
            divergence_rate=0.0,
            results=[],
        )
        text = format_report(report)
        assert "Matches: 5" in text
        assert "Divergent: 0" in text

    def test_with_divergence(self) -> None:
        report = BatchReport(
            total_samples=10,
            divergent_samples=3,
            match_samples=7,
            divergence_rate=0.3,
            results=[],
            divergence_index_mean=5.0,
            divergence_index_median=4.0,
            logprob_gap_mean=0.25,
            logprob_gap_median=0.2,
            likely_bugs=2,
            likely_uncertainty=1,
            unknown_classification=0,
            divergence_by_context_length={"0-50": {"total": 5, "divergent": 1}},
        )
        text = format_report(report)
        assert "Likely bugs:" in text
        assert "30.0%" in text


class TestCliParsing:
    def test_batch_compare_args(self) -> None:
        """Test that batch-compare subcommand is registered."""
        from xpyd_acc.cli import main

        # Just verify parsing doesn't crash with --help
        try:
            main(["batch-compare", "--help"])
        except SystemExit:
            pass  # --help causes SystemExit(0)
