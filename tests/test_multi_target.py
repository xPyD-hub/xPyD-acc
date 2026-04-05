"""Tests for multi-target batch comparison (M32)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from xpyd_acc.batch_compare import (
    DatasetSample,
    MultiTargetBatchReport,
    SampleResult,
    _compute_agreement_matrix,
    compute_report,
    run_multi_batch,
)
from xpyd_acc.cost import TokenUsage


def _make_sample_result(
    sample_id: str,
    baseline_output: str,
    target_output: str,
    *,
    exact_match: bool | None = None,
) -> SampleResult:
    if exact_match is None:
        exact_match = baseline_output == target_output
    return SampleResult(
        sample_id=sample_id,
        prompt="test prompt",
        baseline_output=baseline_output,
        target_output=target_output,
        exact_match=exact_match,
        first_divergence_index=None if exact_match else 0,
        baseline_logprob_at_divergence=None,
        target_logprob_at_divergence=None,
        logprob_gap=None,
        classification="match" if exact_match else "unknown",
        context_length=5,
    )


class TestComputeAgreementMatrix:
    """Tests for _compute_agreement_matrix."""

    def test_identical_targets(self) -> None:
        urls = ["http://t1", "http://t2"]
        r1 = [_make_sample_result("s1", "hello", "hello")]
        r2 = [_make_sample_result("s1", "hello", "hello")]
        per_target = {
            "http://t1": compute_report(r1),
            "http://t2": compute_report(r2),
        }
        matrix = _compute_agreement_matrix(urls, per_target)
        assert matrix["http://t1"]["http://t2"] == 1.0
        assert matrix["http://t1"]["http://t1"] == 1.0

    def test_different_targets(self) -> None:
        urls = ["http://t1", "http://t2"]
        r1 = [_make_sample_result("s1", "hello", "foo")]
        r2 = [_make_sample_result("s1", "hello", "bar")]
        per_target = {
            "http://t1": compute_report(r1),
            "http://t2": compute_report(r2),
        }
        matrix = _compute_agreement_matrix(urls, per_target)
        assert matrix["http://t1"]["http://t2"] == 0.0

    def test_partial_agreement(self) -> None:
        urls = ["http://t1", "http://t2"]
        r1 = [
            _make_sample_result("s1", "hello", "foo"),
            _make_sample_result("s2", "world", "same"),
        ]
        r2 = [
            _make_sample_result("s1", "hello", "bar"),
            _make_sample_result("s2", "world", "same"),
        ]
        per_target = {
            "http://t1": compute_report(r1),
            "http://t2": compute_report(r2),
        }
        matrix = _compute_agreement_matrix(urls, per_target)
        assert matrix["http://t1"]["http://t2"] == 0.5

    def test_empty_results(self) -> None:
        urls = ["http://t1", "http://t2"]
        per_target = {
            "http://t1": compute_report([]),
            "http://t2": compute_report([]),
        }
        matrix = _compute_agreement_matrix(urls, per_target)
        assert matrix["http://t1"]["http://t2"] == 0.0


class TestMultiTargetBatchReport:
    """Tests for MultiTargetBatchReport serialization."""

    def _make_report(self) -> MultiTargetBatchReport:
        r1 = [_make_sample_result("s1", "hello", "hello")]
        r2 = [_make_sample_result("s1", "hello", "diff", exact_match=False)]
        per_target = {
            "http://t1": compute_report(r1),
            "http://t2": compute_report(r2),
        }
        matrix = _compute_agreement_matrix(["http://t1", "http://t2"], per_target)
        return MultiTargetBatchReport(
            baseline_url="http://base",
            target_urls=["http://t1", "http://t2"],
            per_target=per_target,
            agreement_matrix=matrix,
            total_samples=1,
        )

    def test_to_json(self) -> None:
        report = self._make_report()
        data = json.loads(report.to_json())
        assert data["baseline_url"] == "http://base"
        assert len(data["target_urls"]) == 2
        assert "http://t1" in data["per_target"]
        assert "http://t2" in data["per_target"]
        assert data["per_target"]["http://t1"]["divergent_samples"] == 0
        assert data["per_target"]["http://t2"]["divergent_samples"] == 1
        assert "agreement_matrix" in data

    def test_to_markdown(self) -> None:
        report = self._make_report()
        md = report.to_markdown()
        assert "Multi-Target Batch Comparison Report" in md
        assert "Per-Target Summary" in md
        assert "Cross-Target Agreement Matrix" in md
        assert "http://t1" in md
        assert "http://t2" in md

    def test_single_target_no_matrix(self) -> None:
        r1 = [_make_sample_result("s1", "hello", "hello")]
        per_target = {"http://t1": compute_report(r1)}
        matrix = _compute_agreement_matrix(["http://t1"], per_target)
        report = MultiTargetBatchReport(
            baseline_url="http://base",
            target_urls=["http://t1"],
            per_target=per_target,
            agreement_matrix=matrix,
            total_samples=1,
        )
        md = report.to_markdown()
        assert "Cross-Target Agreement Matrix" not in md


class TestRunMultiBatch:
    """Tests for run_multi_batch with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_basic_multi_target(self) -> None:
        samples = [DatasetSample(id="s1", prompt="hello")]

        call_count = 0

        async def mock_collect(url, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if "base" in url:
                return "baseline output", [], "", TokenUsage(), "stop"
            if "t1" in url:
                return "baseline output", [], "", TokenUsage(), "stop"  # matches
            return "different output", [], "", TokenUsage(), "stop"  # diverges

        with patch("xpyd_acc.batch_compare._collect_output", side_effect=mock_collect):
            report = await run_multi_batch(
                samples,
                "http://base",
                ["http://t1", "http://t2"],
            )

        assert report.total_samples == 1
        assert len(report.target_urls) == 2
        assert report.per_target["http://t1"].divergent_samples == 0
        assert report.per_target["http://t2"].divergent_samples == 1
        # Baseline collected once (1 sample), targets collected 2x1
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_progress_callback(self) -> None:
        samples = [
            DatasetSample(id="s1", prompt="a"),
            DatasetSample(id="s2", prompt="b"),
        ]

        async def mock_collect(url, prompt, **kwargs):
            return "output", [], "", TokenUsage(), "stop"

        progress_calls: list[tuple[int, int]] = []

        def on_progress(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        with patch("xpyd_acc.batch_compare._collect_output", side_effect=mock_collect):
            report = await run_multi_batch(
                samples,
                "http://base",
                ["http://t1"],
                on_progress=on_progress,
            )

        assert report.total_samples == 2
        # 2 samples × 1 target = 2 progress calls
        assert len(progress_calls) == 2
        assert progress_calls[-1] == (2, 2)
