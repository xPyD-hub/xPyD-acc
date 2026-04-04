"""Tests for the diagnostic pipeline."""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from xpyd_acc.diagnose import (
    DiagnosticPipeline,
    DiagnosticReport,
    DiagnosticStep,
    StepStatus,
    format_rich_report,
)
from xpyd_acc.logprobs import LogprobsResult, TokenLogprob


def _make_logprobs_result(
    endpoint: str, tokens: list[tuple[str, float]],
) -> LogprobsResult:
    """Helper to build a LogprobsResult."""
    return LogprobsResult(
        endpoint=endpoint,
        model="test-model",
        tokens=[
            TokenLogprob(index=i, token=tok, logprob=lp)
            for i, (tok, lp) in enumerate(tokens)
        ],
    )


def _mock_collect_factory(tokens_a: list[tuple[str, float]], tokens_b: list[tuple[str, float]]):
    """Return a side_effect function that returns different results per endpoint."""
    call_count = 0

    async def _collect(prompt, max_tokens=64, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:  # odd calls = baseline
            return _make_logprobs_result("http://baseline", tokens_a[:max_tokens])
        return _make_logprobs_result("http://target", tokens_b[:max_tokens])

    return _collect


class TestDiagnosticPipeline:
    """Tests for DiagnosticPipeline."""

    @pytest.mark.asyncio
    async def test_all_pass_no_kv(self):
        """All steps pass when endpoints return identical tokens, no KV dumps."""
        tokens = [("Hello", -0.1), ("world", -0.2)]

        pipeline = DiagnosticPipeline(
            baseline_url="http://baseline:8000",
            target_url="http://target:8000",
            prompt="test",
            max_tokens=2,
        )

        mock_collect = _mock_collect_factory(tokens, tokens)
        with patch(
            "xpyd_acc.diagnose.LogprobsCollector.collect",
            side_effect=mock_collect,
        ):
            report = await pipeline.run()

        assert report.overall_pass is True
        assert len(report.steps) == 3
        assert report.steps[0].status == StepStatus.PASS  # first token
        assert report.steps[1].status == StepStatus.SKIP  # KV skipped
        assert report.steps[2].status == StepStatus.PASS  # full sequence

    @pytest.mark.asyncio
    async def test_first_token_divergence(self):
        """First token diverges → step 1 fails."""
        tokens_a = [("Hello", -0.1)]
        tokens_b = [("World", -0.5)]

        pipeline = DiagnosticPipeline(
            baseline_url="http://baseline:8000",
            target_url="http://target:8000",
            prompt="test",
            max_tokens=1,
        )

        mock_collect = _mock_collect_factory(tokens_a, tokens_b)
        with patch(
            "xpyd_acc.diagnose.LogprobsCollector.collect",
            side_effect=mock_collect,
        ):
            report = await pipeline.run()

        assert report.overall_pass is False
        assert report.steps[0].status == StepStatus.FAIL
        assert "Hello" in report.steps[0].detail

    @pytest.mark.asyncio
    async def test_kv_cache_pass(self):
        """KV cache step passes when dumps match."""
        tokens = [("Hi", -0.1)]

        with tempfile.NamedTemporaryFile(suffix=".npz") as f_base, \
             tempfile.NamedTemporaryFile(suffix=".npz") as f_tgt:
            arr = np.random.randn(2, 4, 8).astype(np.float32)
            np.savez(f_base.name, layer_0=arr)
            np.savez(f_tgt.name, layer_0=arr)

            pipeline = DiagnosticPipeline(
                baseline_url="http://baseline:8000",
                target_url="http://target:8000",
                prompt="test",
                max_tokens=1,
                kv_baseline_path=f_base.name,
                kv_target_path=f_tgt.name,
            )

            mock_collect = _mock_collect_factory(tokens, tokens)
            with patch(
                "xpyd_acc.diagnose.LogprobsCollector.collect",
                side_effect=mock_collect,
            ):
                report = await pipeline.run()

        assert report.steps[1].status == StepStatus.PASS
        assert report.steps[1].name == "kv_cache"

    @pytest.mark.asyncio
    async def test_kv_cache_fail(self):
        """KV cache step fails when dumps diverge."""
        tokens = [("Hi", -0.1)]

        with tempfile.NamedTemporaryFile(suffix=".npz") as f_base, \
             tempfile.NamedTemporaryFile(suffix=".npz") as f_tgt:
            arr_a = np.zeros((2, 4, 8), dtype=np.float32)
            arr_b = np.ones((2, 4, 8), dtype=np.float32)
            np.savez(f_base.name, layer_0=arr_a)
            np.savez(f_tgt.name, layer_0=arr_b)

            pipeline = DiagnosticPipeline(
                baseline_url="http://baseline:8000",
                target_url="http://target:8000",
                prompt="test",
                max_tokens=1,
                kv_baseline_path=f_base.name,
                kv_target_path=f_tgt.name,
            )

            mock_collect = _mock_collect_factory(tokens, tokens)
            with patch(
                "xpyd_acc.diagnose.LogprobsCollector.collect",
                side_effect=mock_collect,
            ):
                report = await pipeline.run()

        assert report.overall_pass is False
        assert report.steps[1].status == StepStatus.FAIL
        assert "layer_0" in report.steps[1].detail

    @pytest.mark.asyncio
    async def test_full_sequence_divergence(self):
        """Full sequence diverges at token 2."""
        tokens_a = [("A", -0.1), ("B", -0.2), ("C", -0.3)]
        tokens_b = [("A", -0.1), ("B", -0.2), ("X", -0.9)]

        pipeline = DiagnosticPipeline(
            baseline_url="http://baseline:8000",
            target_url="http://target:8000",
            prompt="test",
            max_tokens=3,
        )

        mock_collect = _mock_collect_factory(tokens_a, tokens_b)
        with patch(
            "xpyd_acc.diagnose.LogprobsCollector.collect",
            side_effect=mock_collect,
        ):
            report = await pipeline.run()

        assert report.overall_pass is False
        # First token should pass (both "A")
        assert report.steps[0].status == StepStatus.PASS
        # Full sequence should fail
        assert report.steps[2].status == StepStatus.FAIL
        assert "token 2" in report.steps[2].detail

    @pytest.mark.asyncio
    async def test_connection_error_handled(self):
        """Network errors are caught and reported as failures."""
        pipeline = DiagnosticPipeline(
            baseline_url="http://nonexistent:9999",
            target_url="http://nonexistent:9999",
            prompt="test",
        )

        # Don't mock — let it fail with connection error
        report = await pipeline.run()

        assert report.overall_pass is False
        assert report.steps[0].status == StepStatus.FAIL
        assert "Error" in report.steps[0].detail

    @pytest.mark.asyncio
    async def test_json_export(self):
        """Report serializes to valid JSON."""
        tokens = [("Hi", -0.1)]

        pipeline = DiagnosticPipeline(
            baseline_url="http://baseline:8000",
            target_url="http://target:8000",
            prompt="test",
            max_tokens=1,
        )

        mock_collect = _mock_collect_factory(tokens, tokens)
        with patch(
            "xpyd_acc.diagnose.LogprobsCollector.collect",
            side_effect=mock_collect,
        ):
            report = await pipeline.run()

        import json
        data = json.loads(report.to_json())
        assert "steps" in data
        assert "overall_pass" in data
        assert len(data["steps"]) == 3


class TestFormatRichReport:
    """Tests for format_rich_report."""

    def test_all_pass(self):
        """Report with all passing steps."""
        report = DiagnosticReport(
            steps=[
                DiagnosticStep("a", "Step A", StepStatus.PASS, "OK"),
                DiagnosticStep("b", "Step B", StepStatus.PASS, "OK"),
            ],
            overall_pass=True,
        )
        text = format_rich_report(report)
        assert "ALL CHECKS PASSED" in text
        assert "✅" in text

    def test_with_failure(self):
        """Report with a failure."""
        report = DiagnosticReport(
            steps=[
                DiagnosticStep("a", "Step A", StepStatus.PASS, "OK"),
                DiagnosticStep("b", "Step B", StepStatus.FAIL, "Bad"),
            ],
            overall_pass=False,
        )
        text = format_rich_report(report)
        assert "FAILED" in text
        assert "❌" in text

    def test_with_skip(self):
        """Report with a skipped step."""
        report = DiagnosticReport(
            steps=[
                DiagnosticStep("a", "Step A", StepStatus.SKIP, "No data"),
            ],
            overall_pass=True,
        )
        text = format_rich_report(report)
        assert "SKIP" in text
