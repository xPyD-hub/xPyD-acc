"""Tests for token timing analysis module."""

from __future__ import annotations

import pytest

from xpyd_acc.streaming import StreamToken
from xpyd_acc.timing import (
    TimingStats,
    _percentile,
    compare_timing,
    compute_timing_stats,
    format_timing_report,
)


def _make_token(index: int, token: str, ts: float) -> StreamToken:
    return StreamToken(index=index, token=token, timestamp=ts)


class TestPercentile:
    """Test the _percentile helper."""

    def test_single_value(self) -> None:
        assert _percentile([5.0], 50) == 5.0
        assert _percentile([5.0], 99) == 5.0

    def test_two_values(self) -> None:
        assert _percentile([1.0, 3.0], 50) == 2.0

    def test_p50_odd(self) -> None:
        assert _percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_p95(self) -> None:
        data = list(range(1, 101))  # 1..100
        result = _percentile([float(x) for x in data], 95)
        assert result == pytest.approx(95.05, abs=0.1)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            _percentile([], 50)


class TestComputeTimingStats:
    """Test compute_timing_stats."""

    def test_empty_tokens(self) -> None:
        stats = compute_timing_stats([], request_start=100.0)
        assert stats.ttft == 0.0
        assert stats.total_tokens == 0
        assert stats.total_duration == 0.0
        assert stats.itl_values == []
        assert stats.itl_p50 is None
        assert stats.itl_p95 is None
        assert stats.itl_p99 is None
        assert stats.itl_mean is None
        assert stats.itl_min is None
        assert stats.itl_max is None

    def test_single_token(self) -> None:
        tokens = [_make_token(0, "Hello", ts=100.5)]
        stats = compute_timing_stats(tokens, request_start=100.0)
        assert stats.ttft == pytest.approx(0.5)
        assert stats.total_tokens == 1
        assert stats.total_duration == pytest.approx(0.5)
        assert stats.itl_values == []
        assert stats.itl_p50 is None

    def test_multiple_tokens(self) -> None:
        tokens = [
            _make_token(0, "A", ts=10.1),   # TTFT = 0.1s
            _make_token(1, "B", ts=10.15),   # ITL = 0.05s
            _make_token(2, "C", ts=10.25),   # ITL = 0.10s
            _make_token(3, "D", ts=10.30),   # ITL = 0.05s
        ]
        stats = compute_timing_stats(tokens, request_start=10.0)
        assert stats.ttft == pytest.approx(0.1)
        assert stats.total_tokens == 4
        assert stats.total_duration == pytest.approx(0.3)
        assert len(stats.itl_values) == 3
        assert stats.itl_mean == pytest.approx(0.0667, abs=0.001)
        assert stats.itl_min == pytest.approx(0.05, abs=0.001)
        assert stats.itl_max == pytest.approx(0.10, abs=0.001)
        assert stats.itl_p50 is not None


class TestCompareTiming:
    """Test compare_timing."""

    def test_basic_comparison(self) -> None:
        baseline = TimingStats(
            ttft=0.1, total_tokens=10, total_duration=0.5,
            itl_values=[0.04, 0.05, 0.04, 0.05, 0.04, 0.05, 0.04, 0.05, 0.04],
        )
        target = TimingStats(
            ttft=0.2, total_tokens=10, total_duration=0.8,
            itl_values=[0.06, 0.07, 0.06, 0.07, 0.06, 0.07, 0.06, 0.07, 0.06],
        )
        report = compare_timing(baseline, target)
        assert report.ttft_diff == pytest.approx(0.1)
        assert report.ttft_ratio == pytest.approx(2.0)

    def test_zero_baseline_ttft(self) -> None:
        baseline = TimingStats(ttft=0.0, total_tokens=1, total_duration=0.0, itl_values=[])
        target = TimingStats(ttft=0.1, total_tokens=1, total_duration=0.1, itl_values=[])
        report = compare_timing(baseline, target)
        assert report.ttft_ratio == float("inf")


class TestFormatTimingReport:
    """Test format_timing_report."""

    def test_basic_format(self) -> None:
        baseline = TimingStats(
            ttft=0.1, total_tokens=5, total_duration=0.5,
            itl_values=[0.05, 0.10, 0.15, 0.10],
        )
        target = TimingStats(
            ttft=0.2, total_tokens=5, total_duration=0.8,
            itl_values=[0.10, 0.15, 0.20, 0.15],
        )
        report = compare_timing(baseline, target)
        text = format_timing_report(report)
        assert "Token Timing Analysis" in text
        assert "TTFT" in text
        assert "Baseline" in text
        assert "Target" in text
        assert "ITL p50" in text
        assert "ITL p95" in text
        assert "ITL p99" in text

    def test_no_itl_tokens(self) -> None:
        baseline = TimingStats(ttft=0.1, total_tokens=1, total_duration=0.1, itl_values=[])
        target = TimingStats(ttft=0.2, total_tokens=1, total_duration=0.2, itl_values=[])
        report = compare_timing(baseline, target)
        text = format_timing_report(report)
        assert "N/A" in text
