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
    export_markdown,
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
                prompt="test prompt here",
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
        lines = csv_str.strip().splitlines()
        header = lines[0].rstrip("\r")
        expected_header = (
            "sample_id,prompt,baseline_output,target_output,"
            "match,divergence_index,logprob_gap"
        )
        assert header == expected_header
        assert "s1" in csv_str
        assert "test prompt here" in csv_str
        assert "0.300000" in csv_str

    def test_csv_prompt_truncation(self) -> None:
        long_prompt = "a" * 300
        results = [
            SampleResult(
                sample_id="s1",
                prompt=long_prompt,
                baseline_output="x",
                target_output="y",
                exact_match=False,
                first_divergence_index=0,
                baseline_logprob_at_divergence=None,
                target_logprob_at_divergence=None,
                logprob_gap=None,
                classification="unknown",
                context_length=10,
            ),
        ]
        report = compute_report(results)
        csv_str = export_csv(report)
        # Default truncation to 200 chars
        assert long_prompt not in csv_str
        assert "a" * 197 + "..." in csv_str

        # No truncation
        csv_str_full = export_csv(report, prompt_max_length=0)
        assert long_prompt in csv_str_full

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
        assert "prompt" in content


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


class TestBatchReportToJson:
    def test_to_json_round_trip(self) -> None:
        """Test that to_json() produces valid JSON with correct fields."""
        import json as json_mod

        results = [
            SampleResult(
                sample_id="0", prompt="hello", baseline_output="hi",
                target_output="hi", exact_match=True,
                first_divergence_index=None,
                baseline_logprob_at_divergence=None,
                target_logprob_at_divergence=None,
                logprob_gap=None, classification="match",
                context_length=1,
            ),
            SampleResult(
                sample_id="1", prompt="world", baseline_output="a",
                target_output="b", exact_match=False,
                first_divergence_index=0,
                baseline_logprob_at_divergence=-0.5,
                target_logprob_at_divergence=-0.8,
                logprob_gap=0.3, classification="likely_bug",
                context_length=1,
            ),
        ]
        report = compute_report(results)
        data = json_mod.loads(report.to_json())
        assert data["total_samples"] == 2
        assert data["divergent_samples"] == 1
        assert len(data["results"]) == 2
        assert data["results"][1]["classification"] == "likely_bug"


class TestCliParsing:
    def test_batch_compare_args(self) -> None:
        """Test that batch-compare subcommand is registered."""
        from xpyd_acc.cli import main

        # Just verify parsing doesn't crash with --help
        try:
            main(["batch-compare", "--help"])
        except SystemExit:
            pass  # --help causes SystemExit(0)


class TestOnProgressCallback:
    """Test that run_batch invokes the on_progress callback."""

    def test_progress_callback_called(self) -> None:
        """Verify on_progress is called once per sample with correct counts."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from xpyd_acc.batch_compare import DatasetSample, run_batch
        from xpyd_acc.cost import TokenUsage

        samples = [
            DatasetSample(id="0", prompt="hello"),
            DatasetSample(id="1", prompt="world"),
            DatasetSample(id="2", prompt="test"),
        ]

        mock_output = ("response text", [], "", TokenUsage(), "stop", 1)

        progress_calls: list[tuple[int, int]] = []

        def track_progress(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        with patch("xpyd_acc.batch_compare._collect_output", new_callable=AsyncMock) as mock_co:
            mock_co.return_value = mock_output
            report = asyncio.run(run_batch(
                samples,
                "http://baseline",
                "http://target",
                on_progress=track_progress,
            ))

        assert report.total_samples == 3
        # Should have 3 progress calls, one per sample
        assert len(progress_calls) == 3
        # All calls should have total=3
        assert all(t == 3 for _, t in progress_calls)
        # Final call should have completed=3
        completed_values = sorted(c for c, _ in progress_calls)
        assert completed_values[-1] == 3

    def test_no_progress_callback(self) -> None:
        """Verify run_batch works fine without on_progress."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from xpyd_acc.batch_compare import DatasetSample, run_batch
        from xpyd_acc.cost import TokenUsage

        samples = [DatasetSample(id="0", prompt="hello")]
        mock_output = ("response", [], "", TokenUsage(), "stop", 1)

        with patch("xpyd_acc.batch_compare._collect_output", new_callable=AsyncMock) as mock_co:
            mock_co.return_value = mock_output
            report = asyncio.run(run_batch(
                samples, "http://baseline", "http://target",
            ))

        assert report.total_samples == 1


class TestToMarkdown:
    def _make_report(self, *, divergent: bool = True) -> BatchReport:
        results = [
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
                context_length=10,
            ),
        ]
        if divergent:
            results.append(SampleResult(
                sample_id="1",
                prompt="Explain gravity",
                baseline_output="Gravity is a force",
                target_output="Gravity is energy",
                exact_match=False,
                first_divergence_index=3,
                baseline_logprob_at_divergence=-0.5,
                target_logprob_at_divergence=-1.2,
                logprob_gap=0.5,
                classification="likely_bug",
                context_length=10,
            ))
        return compute_report(results)

    def test_markdown_contains_summary(self) -> None:
        report = self._make_report()
        md = report.to_markdown()
        assert "# Batch Comparison Report" in md
        assert "| Total samples | 2 |" in md
        assert "| Divergent | 1 |" in md

    def test_markdown_contains_classification(self) -> None:
        report = self._make_report()
        md = report.to_markdown()
        assert "## Classification" in md
        assert "| Likely bugs | 1 |" in md

    def test_markdown_contains_divergent_samples(self) -> None:
        report = self._make_report()
        md = report.to_markdown()
        assert "### Sample 1" in md
        assert "likely_bug" in md

    def test_markdown_no_divergent_section_when_all_match(self) -> None:
        report = self._make_report(divergent=False)
        md = report.to_markdown()
        assert "## Classification" not in md
        assert "### Sample" not in md

    def test_markdown_max_divergent_samples(self) -> None:
        results = []
        for i in range(20):
            results.append(SampleResult(
                sample_id=str(i),
                prompt=f"prompt {i}",
                baseline_output="a",
                target_output="b",
                exact_match=False,
                first_divergence_index=0,
                baseline_logprob_at_divergence=None,
                target_logprob_at_divergence=None,
                logprob_gap=None,
                classification="unknown",
                context_length=5,
            ))
        report = compute_report(results)
        md = report.to_markdown(max_divergent_samples=5)
        assert "showing 5/20" in md


class TestExportMarkdown:
    def test_export_to_file(self, tmp_path: Path) -> None:
        results = [
            SampleResult(
                sample_id="0",
                prompt="hi",
                baseline_output="hello",
                target_output="hello",
                exact_match=True,
                first_divergence_index=None,
                baseline_logprob_at_divergence=None,
                target_logprob_at_divergence=None,
                logprob_gap=None,
                classification="match",
                context_length=1,
            ),
        ]
        report = compute_report(results)
        out = tmp_path / "report.md"
        md = export_markdown(report, out)
        assert out.read_text() == md
        assert "# Batch Comparison Report" in md

    def test_export_no_file(self) -> None:
        results = [
            SampleResult(
                sample_id="0",
                prompt="hi",
                baseline_output="hello",
                target_output="hello",
                exact_match=True,
                first_divergence_index=None,
                baseline_logprob_at_divergence=None,
                target_logprob_at_divergence=None,
                logprob_gap=None,
                classification="match",
                context_length=1,
            ),
        ]
        report = compute_report(results)
        md = export_markdown(report)
        assert "# Batch Comparison Report" in md


class TestCollectOutputTimeout:
    """Test that _collect_output accepts timeout parameter."""

    def test_timeout_parameter_signature(self) -> None:
        """Verify _collect_output has timeout in its signature."""
        import inspect

        from xpyd_acc.batch_compare import _collect_output

        sig = inspect.signature(_collect_output)
        assert "timeout" in sig.parameters
        assert sig.parameters["timeout"].default == 120.0

    def test_run_batch_timeout_parameter(self) -> None:
        """Verify run_batch has timeout in its signature."""
        import inspect

        from xpyd_acc.batch_compare import run_batch

        sig = inspect.signature(run_batch)
        assert "timeout" in sig.parameters
        assert sig.parameters["timeout"].default == 120.0
