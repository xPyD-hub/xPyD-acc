"""Tests for multi-dataset batch comparison."""

from __future__ import annotations

import json

import pytest

from xpyd_acc.batch_compare import BatchReport, DatasetSample, SampleResult
from xpyd_acc.multi_dataset import (
    MultiDatasetReport,
    format_multi_dataset_report,
    run_multi_dataset,
)


def _make_result(sample_id: str, match: bool) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        prompt="test prompt",
        baseline_output="hello world",
        target_output="hello world" if match else "hello moon",
        exact_match=match,
        first_divergence_index=None if match else 1,
        baseline_logprob_at_divergence=None if match else -0.1,
        target_logprob_at_divergence=None if match else -0.6,
        logprob_gap=None if match else 0.5,
        classification="match" if match else "likely_bug",
        context_length=10,
    )


def _make_report(
    name: str,
    results: list[SampleResult],
    baseline_url: str = "http://base",
    target_url: str = "http://target",
) -> BatchReport:
    matched = sum(1 for r in results if r.exact_match)
    divergent = len(results) - matched
    rate = divergent / len(results) if results else 0.0
    return BatchReport(
        total_samples=len(results),
        match_samples=matched,
        divergent_samples=divergent,
        divergence_rate=rate,
        results=results,
    )


class TestMultiDatasetReport:
    """Tests for MultiDatasetReport dataclass."""

    def test_basic_construction(self):
        r1 = _make_report("ds1", [_make_result("s1", True), _make_result("s2", False)])
        r2 = _make_report("ds2", [_make_result("s3", True)])

        report = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="test-model",
            datasets=["ds1", "ds2"],
            per_dataset={"ds1": r1, "ds2": r2},
        )

        assert report.total_samples == 3
        assert report.total_divergent == 1
        assert abs(report.overall_divergence_rate - 1 / 3) < 0.01

    def test_all_match(self):
        r1 = _make_report("ds1", [_make_result("s1", True)])
        r2 = _make_report("ds2", [_make_result("s2", True)])

        report = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["ds1", "ds2"],
            per_dataset={"ds1": r1, "ds2": r2},
        )

        assert report.total_divergent == 0
        assert report.overall_divergence_rate == 0.0

    def test_all_divergent(self):
        r1 = _make_report("ds1", [_make_result("s1", False)])
        r2 = _make_report("ds2", [_make_result("s2", False)])

        report = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["ds1", "ds2"],
            per_dataset={"ds1": r1, "ds2": r2},
        )

        assert report.total_divergent == 2
        assert report.overall_divergence_rate == 1.0

    def test_empty_datasets(self):
        report = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=[],
            per_dataset={},
        )

        assert report.total_samples == 0
        assert report.overall_divergence_rate == 0.0

    def test_to_json_roundtrip(self):
        r1 = _make_report("ds1", [_make_result("s1", True), _make_result("s2", False)])
        report = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="test-model",
            datasets=["ds1"],
            per_dataset={"ds1": r1},
        )

        data = json.loads(report.to_json())
        assert data["baseline_url"] == "http://base"
        assert data["target_url"] == "http://target"
        assert data["model"] == "test-model"
        assert data["total_samples"] == 2
        assert data["total_divergent"] == 1
        assert "ds1" in data["per_dataset"]
        assert data["per_dataset"]["ds1"]["total_samples"] == 2

    def test_to_markdown(self):
        r1 = _make_report("ds1", [_make_result("s1", True), _make_result("s2", False)])
        r2 = _make_report("ds2", [_make_result("s3", True)])
        report = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="test-model",
            datasets=["ds1", "ds2"],
            per_dataset={"ds1": r1, "ds2": r2},
        )

        md = report.to_markdown()
        assert "# Multi-Dataset Batch Comparison Report" in md
        assert "ds1" in md
        assert "ds2" in md
        assert "Per-Dataset Summary" in md
        assert "50.0%" in md  # ds1 divergence rate

    def test_single_dataset(self):
        r1 = _make_report("only", [_make_result("s1", False)])
        report = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["only"],
            per_dataset={"only": r1},
        )
        assert report.total_samples == 1
        assert report.overall_divergence_rate == 1.0


class TestFormatMultiDatasetReport:
    """Tests for terminal formatting."""

    def test_format_basic(self):
        r1 = _make_report("gsm8k", [_make_result("s1", True)])
        r2 = _make_report("mmlu", [_make_result("s2", False)])

        report = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="test-model",
            datasets=["gsm8k", "mmlu"],
            per_dataset={"gsm8k": r1, "mmlu": r2},
        )

        text = format_multi_dataset_report(report)
        assert "gsm8k" in text
        assert "mmlu" in text
        assert "✓" in text  # gsm8k passes
        assert "✗" in text  # mmlu fails
        assert "50.0%" in text  # overall rate

    def test_format_all_pass(self):
        r1 = _make_report("ds1", [_make_result("s1", True)])
        report = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["ds1"],
            per_dataset={"ds1": r1},
        )

        text = format_multi_dataset_report(report)
        assert "✓" in text
        assert "✗" not in text


class TestRunMultiDataset:
    """Tests for async run_multi_dataset function."""

    @pytest.mark.asyncio
    async def test_run_with_mock(self, monkeypatch):
        """Test multi-dataset run with mocked run_batch."""
        call_log = []

        async def mock_run_batch(samples, baseline_url, target_url, **kwargs):
            call_log.append(len(samples))
            results = [_make_result(s.id, i % 2 == 0) for i, s in enumerate(samples)]
            return _make_report("test", results, baseline_url, target_url)

        monkeypatch.setattr("xpyd_acc.multi_dataset.run_batch", mock_run_batch)

        dataset_map = {
            "ds1": [DatasetSample(id="s1", prompt="p1"), DatasetSample(id="s2", prompt="p2")],
            "ds2": [DatasetSample(id="s3", prompt="p3")],
        }

        report = await run_multi_dataset(
            dataset_map,
            "http://base",
            "http://target",
        )

        assert len(call_log) == 2
        assert set(call_log) == {1, 2}
        assert report.total_samples == 3
        assert len(report.datasets) == 2
        assert "ds1" in report.per_dataset
        assert "ds2" in report.per_dataset

    @pytest.mark.asyncio
    async def test_callback_invoked(self, monkeypatch):
        """Test on_dataset_complete callback is called."""
        async def mock_run_batch(samples, baseline_url, target_url, **kwargs):
            results = [_make_result(s.id, True) for s in samples]
            return _make_report("test", results, baseline_url, target_url)

        monkeypatch.setattr("xpyd_acc.multi_dataset.run_batch", mock_run_batch)

        completed = []

        def on_complete(name, report):
            completed.append(name)

        dataset_map = {
            "ds1": [DatasetSample(id="s1", prompt="p1")],
            "ds2": [DatasetSample(id="s2", prompt="p2")],
        }

        await run_multi_dataset(
            dataset_map,
            "http://base",
            "http://target",
            on_dataset_complete=on_complete,
        )

        assert set(completed) == {"ds1", "ds2"}

    @pytest.mark.asyncio
    async def test_empty_dataset_map(self, monkeypatch):
        """Test with no datasets."""
        async def mock_run_batch(samples, baseline_url, target_url, **kwargs):
            raise AssertionError("should not be called")

        monkeypatch.setattr("xpyd_acc.multi_dataset.run_batch", mock_run_batch)

        report = await run_multi_dataset({}, "http://base", "http://target")

        assert report.total_samples == 0
        assert report.overall_divergence_rate == 0.0
        assert report.datasets == []

    @pytest.mark.asyncio
    async def test_kwargs_forwarded(self, monkeypatch):
        """Test that kwargs are forwarded to run_batch."""
        received_kwargs = {}

        async def mock_run_batch(samples, baseline_url, target_url, **kwargs):
            received_kwargs.update(kwargs)
            results = [_make_result(s.id, True) for s in samples]
            return _make_report("test", results, baseline_url, target_url)

        monkeypatch.setattr("xpyd_acc.multi_dataset.run_batch", mock_run_batch)

        dataset_map = {"ds1": [DatasetSample(id="s1", prompt="p1")]}

        await run_multi_dataset(
            dataset_map,
            "http://base",
            "http://target",
            model="gpt-4",
            max_tokens=128,
            timeout=30.0,
        )

        assert received_kwargs["model"] == "gpt-4"
        assert received_kwargs["max_tokens"] == 128
        assert received_kwargs["timeout"] == 30.0


class TestMultiDatasetMarkdownDetails:
    """Test markdown output edge cases."""

    def test_markdown_truncates_divergent(self):
        """Test that markdown limits divergent samples shown."""
        results = [_make_result(f"s{i}", False) for i in range(15)]
        r1 = _make_report("ds1", results)
        report = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["ds1"],
            per_dataset={"ds1": r1},
        )

        md = report.to_markdown(max_divergent_samples=5)
        assert "... and 10 more" in md

    def test_json_has_all_fields(self):
        """Test JSON export contains all expected fields."""
        r1 = _make_report("ds1", [_make_result("s1", True)])
        report = MultiDatasetReport(
            baseline_url="http://base",
            target_url="http://target",
            model="m",
            datasets=["ds1"],
            per_dataset={"ds1": r1},
        )

        data = json.loads(report.to_json())
        assert "baseline_url" in data
        assert "target_url" in data
        assert "model" in data
        assert "datasets" in data
        assert "total_samples" in data
        assert "total_divergent" in data
        assert "overall_divergence_rate" in data
        assert "per_dataset" in data
