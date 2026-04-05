"""Tests for multi-model comparison."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from xpyd_acc.batch_compare import BatchReport, DatasetSample, SampleResult
from xpyd_acc.multi_model import (
    CrossModelSummary,
    MultiModelBatchReport,
    compute_cross_model_summary,
    format_multi_model_report,
    run_multi_model,
)


def _make_result(sample_id: str, match: bool) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        prompt=f"prompt-{sample_id}",
        baseline_output="out",
        target_output="out" if match else "diff",
        exact_match=match,
        first_divergence_index=None if match else 0,
        baseline_logprob_at_divergence=None,
        target_logprob_at_divergence=None,
        logprob_gap=None if match else 0.5,
        classification="match" if match else "likely_bug",
        context_length=10,
    )


def _make_report(results: list[SampleResult]) -> BatchReport:
    divergent = sum(1 for r in results if r.is_divergent())
    total = len(results)
    return BatchReport(
        total_samples=total,
        divergent_samples=divergent,
        match_samples=total - divergent,
        divergence_rate=divergent / total if total else 0.0,
        results=results,
    )


class TestCrossModelSummary:
    def test_to_dict(self) -> None:
        s = CrossModelSummary(
            per_sample_divergent_models={"s1": ["m1"], "s2": ["m1", "m2"]},
            systematic_divergent_ids=["s2"],
            model_specific_divergent_ids=["s1"],
            all_match_ids=[],
        )
        d = s.to_dict()
        assert d["systematic_divergent_count"] == 1
        assert d["model_specific_divergent_count"] == 1
        assert d["all_match_count"] == 0

    def test_empty(self) -> None:
        s = CrossModelSummary()
        d = s.to_dict()
        assert d["systematic_divergent_count"] == 0


class TestComputeCrossModelSummary:
    def test_all_match(self) -> None:
        r1 = _make_report([_make_result("s1", True), _make_result("s2", True)])
        r2 = _make_report([_make_result("s1", True), _make_result("s2", True)])
        cm = compute_cross_model_summary(["m1", "m2"], {"m1": r1, "m2": r2})
        assert cm.all_match_ids == ["s1", "s2"]
        assert cm.systematic_divergent_ids == []
        assert cm.model_specific_divergent_ids == []

    def test_systematic_divergence(self) -> None:
        r1 = _make_report([_make_result("s1", False), _make_result("s2", True)])
        r2 = _make_report([_make_result("s1", False), _make_result("s2", True)])
        cm = compute_cross_model_summary(["m1", "m2"], {"m1": r1, "m2": r2})
        assert cm.systematic_divergent_ids == ["s1"]
        assert cm.all_match_ids == ["s2"]

    def test_model_specific_divergence(self) -> None:
        r1 = _make_report([_make_result("s1", False), _make_result("s2", True)])
        r2 = _make_report([_make_result("s1", True), _make_result("s2", True)])
        cm = compute_cross_model_summary(["m1", "m2"], {"m1": r1, "m2": r2})
        assert cm.model_specific_divergent_ids == ["s1"]
        assert cm.systematic_divergent_ids == []

    def test_empty_models(self) -> None:
        cm = compute_cross_model_summary([], {})
        assert cm.all_match_ids == []

    def test_single_model(self) -> None:
        r1 = _make_report([_make_result("s1", False)])
        cm = compute_cross_model_summary(["m1"], {"m1": r1})
        assert cm.systematic_divergent_ids == ["s1"]
        assert cm.model_specific_divergent_ids == []


class TestMultiModelBatchReport:
    def _make_report(self) -> MultiModelBatchReport:
        r1 = _make_report([_make_result("s1", True), _make_result("s2", False)])
        r2 = _make_report([_make_result("s1", False), _make_result("s2", False)])
        cm = compute_cross_model_summary(["m1", "m2"], {"m1": r1, "m2": r2})
        return MultiModelBatchReport(
            baseline_url="http://base",
            target_url="http://target",
            models=["m1", "m2"],
            per_model={"m1": r1, "m2": r2},
            cross_model=cm,
            total_samples=2,
        )

    def test_to_json_roundtrip(self) -> None:
        report = self._make_report()
        data = json.loads(report.to_json())
        assert data["models"] == ["m1", "m2"]
        assert "m1" in data["per_model"]
        assert "m2" in data["per_model"]
        assert data["cross_model"]["systematic_divergent_count"] == 1
        assert data["cross_model"]["model_specific_divergent_count"] == 1

    def test_to_markdown(self) -> None:
        report = self._make_report()
        md = report.to_markdown()
        assert "Multi-Model" in md
        assert "`m1`" in md
        assert "`m2`" in md
        assert "Systematic" in md

    def test_to_markdown_no_systematic(self) -> None:
        r1 = _make_report([_make_result("s1", True)])
        r2 = _make_report([_make_result("s1", True)])
        cm = compute_cross_model_summary(["m1", "m2"], {"m1": r1, "m2": r2})
        report = MultiModelBatchReport(
            baseline_url="http://b",
            target_url="http://t",
            models=["m1", "m2"],
            per_model={"m1": r1, "m2": r2},
            cross_model=cm,
            total_samples=1,
        )
        md = report.to_markdown()
        assert "Systematic Divergences" not in md


class TestFormatMultiModelReport:
    def test_format(self) -> None:
        r1 = _make_report([_make_result("s1", True)])
        r2 = _make_report([_make_result("s1", False)])
        cm = compute_cross_model_summary(["m1", "m2"], {"m1": r1, "m2": r2})
        report = MultiModelBatchReport(
            baseline_url="http://b",
            target_url="http://t",
            models=["m1", "m2"],
            per_model={"m1": r1, "m2": r2},
            cross_model=cm,
            total_samples=1,
        )
        output = format_multi_model_report(report)
        assert "✅ m1" in output
        assert "❌ m2" in output
        assert "Model-specific" in output


class TestRunMultiModel:
    @pytest.mark.asyncio
    async def test_runs_each_model(self) -> None:
        samples = [DatasetSample(id="s1", prompt="hello")]
        mock_report = _make_report([_make_result("s1", True)])

        completed_models: list[str] = []

        def on_complete(model: str, report: BatchReport) -> None:
            completed_models.append(model)

        with patch("xpyd_acc.multi_model.run_batch", new_callable=AsyncMock) as mock_rb:
            mock_rb.return_value = mock_report
            result = await run_multi_model(
                samples,
                "http://b",
                "http://t",
                ["m1", "m2", "m3"],
                on_model_complete=on_complete,
            )

        assert mock_rb.call_count == 3
        assert result.models == ["m1", "m2", "m3"]
        assert len(result.per_model) == 3
        assert completed_models == ["m1", "m2", "m3"]
        assert result.total_samples == 1

    @pytest.mark.asyncio
    async def test_single_model_backward_compat(self) -> None:
        samples = [DatasetSample(id="s1", prompt="hello")]
        mock_report = _make_report([_make_result("s1", False)])

        with patch("xpyd_acc.multi_model.run_batch", new_callable=AsyncMock) as mock_rb:
            mock_rb.return_value = mock_report
            result = await run_multi_model(
                samples, "http://b", "http://t", ["single-model"]
            )

        assert mock_rb.call_count == 1
        assert result.models == ["single-model"]
        assert result.cross_model.systematic_divergent_ids == ["s1"]

    @pytest.mark.asyncio
    async def test_passes_model_to_run_batch(self) -> None:
        samples = [DatasetSample(id="s1", prompt="hello")]
        mock_report = _make_report([_make_result("s1", True)])

        with patch("xpyd_acc.multi_model.run_batch", new_callable=AsyncMock) as mock_rb:
            mock_rb.return_value = mock_report
            await run_multi_model(
                samples, "http://b", "http://t", ["mymodel"],
                max_tokens=128, api_key="test-key",
            )

        kw = mock_rb.call_args.kwargs
        assert kw.get("model") == "mymodel"

    @pytest.mark.asyncio
    async def test_cross_model_with_mixed_results(self) -> None:
        samples = [
            DatasetSample(id="s1", prompt="p1"),
            DatasetSample(id="s2", prompt="p2"),
            DatasetSample(id="s3", prompt="p3"),
        ]
        # m1: s1 diverges, s2 diverges, s3 matches
        report_m1 = _make_report([
            _make_result("s1", False),
            _make_result("s2", False),
            _make_result("s3", True),
        ])
        # m2: s1 diverges, s2 matches, s3 matches
        report_m2 = _make_report([
            _make_result("s1", False),
            _make_result("s2", True),
            _make_result("s3", True),
        ])

        call_count = 0

        async def fake_run_batch(samps, base, target, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("model") == "m1":
                return report_m1
            return report_m2

        with patch("xpyd_acc.multi_model.run_batch", side_effect=fake_run_batch):
            result = await run_multi_model(
                samples, "http://b", "http://t", ["m1", "m2"]
            )

        assert result.cross_model.systematic_divergent_ids == ["s1"]
        assert result.cross_model.model_specific_divergent_ids == ["s2"]
        assert result.cross_model.all_match_ids == ["s3"]
