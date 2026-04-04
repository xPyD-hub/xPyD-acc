"""Tests for watch mode."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xpyd_acc.logprobs import ComparisonReport, DivergencePoint, LogprobsResult, TokenLogprob
from xpyd_acc.watch import WatchIteration, _build_table, run_watch


def _make_logprobs_result(tokens: list[str], endpoint: str = "http://test") -> LogprobsResult:
    """Helper to create LogprobsResult."""
    return LogprobsResult(
        endpoint=endpoint,
        model="test-model",
        tokens=[TokenLogprob(index=i, token=t, logprob=-0.1) for i, t in enumerate(tokens)],
    )


def _matching_report() -> ComparisonReport:
    baseline = _make_logprobs_result(["Hello", " world"], "http://baseline")
    target = _make_logprobs_result(["Hello", " world"], "http://target")
    return ComparisonReport(
        baseline=baseline,
        target=target,
        divergence=None,
        total_tokens_compared=2,
        match=True,
    )


def _divergent_report() -> ComparisonReport:
    baseline = _make_logprobs_result(["Hello", " world"], "http://baseline")
    target = _make_logprobs_result(["Hello", " earth"], "http://target")
    return ComparisonReport(
        baseline=baseline,
        target=target,
        divergence=DivergencePoint(
            token_index=1,
            expected_token=" world",
            actual_token=" earth",
            expected_logprob=-0.1,
            actual_logprob=-0.2,
            prob_diff=0.1,
        ),
        total_tokens_compared=2,
        match=False,
    )


class TestBuildTable:
    """Tests for _build_table."""

    def test_empty(self) -> None:
        table = _build_table([], {"total": 0, "passed": 0})
        assert table.title == "xpyd-acc watch"

    def test_with_iterations(self) -> None:
        iterations = [
            WatchIteration(1, 1000.0, True, None, None, None, 0.5),
            WatchIteration(2, 1060.0, False, 5, "a", "b", 0.8),
        ]
        table = _build_table(iterations, {"total": 2, "passed": 1})
        assert table.row_count == 3  # 2 rows + 1 summary


class TestRunWatch:
    """Tests for run_watch."""

    @pytest.mark.asyncio
    async def test_max_iterations(self) -> None:
        """Watch stops after max_iterations."""
        with patch("xpyd_acc.watch.LogprobsCollector") as MockCollector, \
             patch("xpyd_acc.watch.LogprobsComparator") as MockComparator:
            mock_collector_instance = MagicMock()
            mock_collector_instance.collect = AsyncMock(
                return_value=_make_logprobs_result(["a", "b"])
            )
            MockCollector.return_value = mock_collector_instance

            mock_comparator_instance = MagicMock()
            mock_comparator_instance.compare.return_value = _matching_report()
            MockComparator.return_value = mock_comparator_instance

            summary = await run_watch(
                baseline_url="http://baseline",
                target_url="http://target",
                prompt="test",
                max_iterations=3,
                interval=0.01,
                no_live=True,
            )

            assert summary.total_iterations == 3
            assert summary.passed == 3
            assert summary.failed == 0
            assert summary.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_alert_threshold(self) -> None:
        """Watch exits after alert_threshold consecutive failures."""
        with patch("xpyd_acc.watch.LogprobsCollector") as MockCollector, \
             patch("xpyd_acc.watch.LogprobsComparator") as MockComparator:
            mock_collector_instance = MagicMock()
            mock_collector_instance.collect = AsyncMock(
                return_value=_make_logprobs_result(["a", "b"])
            )
            MockCollector.return_value = mock_collector_instance

            mock_comparator_instance = MagicMock()
            mock_comparator_instance.compare.return_value = _divergent_report()
            MockComparator.return_value = mock_comparator_instance

            summary = await run_watch(
                baseline_url="http://baseline",
                target_url="http://target",
                prompt="test",
                max_iterations=10,
                alert_threshold=2,
                interval=0.01,
                no_live=True,
            )

            assert summary.total_iterations == 2
            assert summary.failed == 2
            assert summary.consecutive_failures_at_end == 2

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """Errors are captured without crashing the loop."""
        with patch("xpyd_acc.watch.LogprobsCollector") as MockCollector, \
             patch("xpyd_acc.watch.LogprobsComparator"):
            mock_collector_instance = MagicMock()
            mock_collector_instance.collect = AsyncMock(side_effect=ConnectionError("down"))
            MockCollector.return_value = mock_collector_instance

            summary = await run_watch(
                baseline_url="http://baseline",
                target_url="http://target",
                prompt="test",
                max_iterations=2,
                interval=0.01,
                no_live=True,
            )

            assert summary.total_iterations == 2
            assert summary.errors == 2
            assert all(it.error == "down" for it in summary.iterations)

    @pytest.mark.asyncio
    async def test_json_log(self) -> None:
        """JSON log file is written correctly."""
        with patch("xpyd_acc.watch.LogprobsCollector") as MockCollector, \
             patch("xpyd_acc.watch.LogprobsComparator") as MockComparator:
            mock_collector_instance = MagicMock()
            mock_collector_instance.collect = AsyncMock(
                return_value=_make_logprobs_result(["a"])
            )
            MockCollector.return_value = mock_collector_instance

            mock_comparator_instance = MagicMock()
            mock_comparator_instance.compare.return_value = _matching_report()
            MockComparator.return_value = mock_comparator_instance

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                log_path = f.name

            await run_watch(
                baseline_url="http://baseline",
                target_url="http://target",
                prompt="test",
                max_iterations=3,
                interval=0.01,
                log_path=log_path,
                no_live=True,
            )

            log_data = json.loads(Path(log_path).read_text())
            assert len(log_data) == 3
            assert all(entry["passed"] for entry in log_data)
            Path(log_path).unlink()

    @pytest.mark.asyncio
    async def test_mixed_results(self) -> None:
        """Mix of pass and fail iterations."""
        call_count = 0

        with patch("xpyd_acc.watch.LogprobsCollector") as MockCollector, \
             patch("xpyd_acc.watch.LogprobsComparator") as MockComparator:
            mock_collector_instance = MagicMock()
            mock_collector_instance.collect = AsyncMock(
                return_value=_make_logprobs_result(["a", "b"])
            )
            MockCollector.return_value = mock_collector_instance

            def compare_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    return _divergent_report()
                return _matching_report()

            mock_comparator_instance = MagicMock()
            mock_comparator_instance.compare.side_effect = compare_side_effect
            MockComparator.return_value = mock_comparator_instance

            summary = await run_watch(
                baseline_url="http://baseline",
                target_url="http://target",
                prompt="test",
                max_iterations=4,
                interval=0.01,
                no_live=True,
            )

            assert summary.total_iterations == 4
            assert summary.passed == 2
            assert summary.failed == 2
            assert summary.pass_rate == 0.5
